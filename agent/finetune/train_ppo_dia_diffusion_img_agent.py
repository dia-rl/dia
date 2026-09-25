"""DIA fine-tuning for pixel observations."""

import os
import math
import time
import numpy as np
import torch
import logging
import einops

log = logging.getLogger(__name__)

from agent.finetune.train_ppo_dia_diffusion_agent import TrainPPODIADiffusionAgent
from model.common.modules import RandomShiftsAug


class TrainPPODIAImgDiffusionAgent(TrainPPODIADiffusionAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Image randomization
        self.augment = cfg.train.augment
        if self.augment:
            self.aug = RandomShiftsAug(pad=4)

        # obs is a dict of modalities rather than a single state array
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}

        # Gradient accumulation to deal with large GPU RAM usage
        self.grad_accumulate = cfg.train.grad_accumulate

        # Encoding chunk for the no-grad passes over the whole rollout
        self.img_encode_chunk = int(cfg.train.get("img_encode_chunk", 256))

    def _obs_chunks(self, obs_d, size):
        """Yield (start, end, {state, rgb}) slices of a flattened observation dict."""
        n = obs_d["state"].shape[0]
        for st in range(0, n, size):
            ed = min(st + size, n)
            yield st, ed, {k: obs_d[k][st:ed] for k in obs_d}

    def run(self):
        timer_start = time.time()
        resume_itr = int(self.cfg.train.get("resume_from_itr", 0))
        if resume_itr > 0:
            self.load(resume_itr)
            self.itr = resume_itr + 1
            log.info(f"Resumed from iter {resume_itr}, starting at iter {self.itr}")
        else:
            self.itr = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        prev_obs_venv = None

        while self.itr < self.n_train_itr:
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()

            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if (
                self.reset_at_iteration
                or eval_mode
                or last_itr_eval
                or prev_obs_venv is None
            ):
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv
            last_itr_eval = eval_mode

            obs_trajs = {
                k: np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dims[k])
                )
                for k in self.obs_dims
            }
            next_obs_trajs = {
                k: np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dims[k])
                )
                for k in self.obs_dims
            }
            K_ft = self.model.ft_denoising_steps
            chains_trajs = np.zeros(
                (self.n_steps, self.n_envs, K_ft + 1, self.horizon_steps, self.action_dim)
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            # ---------- Rollout ----------
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")
                with torch.no_grad():
                    cond = {
                        key: torch.from_numpy(prev_obs_venv[key]).float().to(self.device)
                        for key in self.obs_dims
                    }
                    samples = self.model(cond=cond, deterministic=eval_mode, return_chain=True)
                    output_venv = samples.trajectories.cpu().numpy()
                    chains_venv = samples.chains.cpu().numpy()
                action_venv = output_venv[:, : self.act_steps]

                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                for k in obs_trajs:
                    obs_trajs[k][step] = prev_obs_venv[k]
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                # next obs for the Q target; on truncation the wrapper resets within the
                # step, so the true next obs arrives in info as "final_obs"
                robomimic_info = isinstance(info_venv, (list, tuple))
                for i in range(self.n_envs):
                    if robomimic_info and truncated_venv[i] and "final_obs" in info_venv[i]:
                        for k in next_obs_trajs:
                            next_obs_trajs[k][step, i] = info_venv[i]["final_obs"][k]
                    else:
                        for k in next_obs_trajs:
                            next_obs_trajs[k][step, i] = obs_venv[k][i]
                prev_obs_venv = obs_venv

            # ---------- Episode stats ----------
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    s_, e_ = env_steps[i], env_steps[i + 1]
                    if e_ - s_ > 1:
                        episodes_start_end.append((env_ind, s_, e_ - 1))
            if len(episodes_start_end) > 0:
                rew_split = [
                    reward_trajs[s_ : e_ + 1, env_ind]
                    for env_ind, s_, e_ in episodes_start_end
                ]
                episode_reward = np.array([np.sum(r) for r in rew_split])
                episode_best_reward = np.array(
                    [np.max(r) / self.act_steps for r in rew_split]
                )
                avg_episode_reward = float(np.mean(episode_reward))
                avg_best_reward = float(np.mean(episode_best_reward))
                success_rate = float(
                    np.mean(episode_best_reward >= self.best_reward_threshold_for_success)
                )
            else:
                avg_episode_reward = 0.0
                avg_best_reward = 0.0
                success_rate = 0.0
                log.info("[WARNING] No episode completed within the iteration!")

            # ---------- Update ----------
            critic_loss_avg = 0.0
            critic_q_loss_avg = 0.0
            pg_loss_avg = 0.0
            adv_outer_abs_mean = 0.0
            kl_mean = 0.0
            clipfrac_mean = 0.0
            v_inner_loss_avg = 0.0
            a_inner_abs_mean = 0.0
            a_inner_scaled_mean = 0.0
            v_inner_start_mean = 0.0
            v_inner_final_mean = 0.0

            if not eval_mode:
                S, E = self.n_steps, self.n_envs
                N = S * E

                with torch.no_grad():
                    obs_d = {
                        k: torch.from_numpy(obs_trajs[k]).float().to(self.device)
                        for k in obs_trajs
                    }
                    next_obs_d = {
                        k: torch.from_numpy(next_obs_trajs[k]).float().to(self.device)
                        for k in next_obs_trajs
                    }
                    # image randomization, applied once per iteration as in DPPO's image agent
                    if self.augment:
                        rgb = einops.rearrange(obs_d["rgb"], "s e t c h w -> (s e t) c h w")
                        obs_d["rgb"] = einops.rearrange(
                            self.aug(rgb),
                            "(s e t) c h w -> s e t c h w",
                            s=self.n_steps,
                            e=self.n_envs,
                        )

                    chains_d = torch.from_numpy(chains_trajs).float().to(self.device)
                    actions_d = chains_d[:, :, -1]

                    obs_flat = {k: obs_d[k].reshape(N, *obs_d[k].shape[2:]) for k in obs_d}
                    next_obs_flat = {
                        k: next_obs_d[k].reshape(N, *next_obs_d[k].shape[2:])
                        for k in next_obs_d
                    }

                    # V (state critic over images) per env step, chunked
                    values_list = []
                    for _, _, ob in self._obs_chunks(obs_flat, self.logprob_batch_size):
                        values_list.append(
                            self.model.critic(ob, no_augment=True).view(-1)
                        )
                    values_trajs = torch.cat(values_list).view(S, E).cpu().numpy()

                    obs_venv_ts = {
                        key: torch.from_numpy(obs_venv[key]).float().to(self.device)
                        for key in self.obs_dims
                    }
                    boot_v = (
                        self.model.critic(obs_venv_ts, no_augment=True)
                        .view(-1)
                        .cpu()
                        .numpy()
                    )

                    # Per-K log-probs, chunked
                    chains_flat_d = chains_d.reshape(
                        N, K_ft + 1, self.horizon_steps, self.action_dim
                    )
                    lp_list = []
                    for st, ed, ob in self._obs_chunks(obs_flat, self.logprob_batch_size):
                        lp_list.append(
                            self.model.get_logprobs(ob, chains_flat_d[st:ed]).cpu().numpy()
                        )
                    logprobs_trajs = np.concatenate(lp_list, axis=0).reshape(
                        N, K_ft, self.horizon_steps, self.action_dim
                    )

                # ---------- Running reward scaling ----------
                if self.reward_scale_running:
                    reward_trajs_t = self.running_reward_scaler(
                        reward=reward_trajs.T, first=firsts_trajs[:-1].T
                    )
                    reward_trajs_scaled = reward_trajs_t.T
                else:
                    reward_trajs_scaled = reward_trajs

                # ---------- Outer GAE ----------
                advantages_outer = np.zeros((S, E))
                lastgaelam = np.zeros(E)
                for t in reversed(range(S)):
                    nextv = values_trajs[t + 1] if t < S - 1 else boot_v
                    nonterm = 1.0 - terminated_trajs[t]
                    delta = (
                        reward_trajs_scaled[t] * self.reward_scale_const
                        + self.gamma * nextv * nonterm
                        - values_trajs[t]
                    )
                    advantages_outer[t] = lastgaelam = (
                        delta + self.gamma * self.gae_lambda * nonterm * lastgaelam
                    )
                returns_outer = advantages_outer + values_trajs
                adv_outer_abs_mean = float(np.abs(advantages_outer).mean())

                advantages_2d = advantages_outer[:, :, None] + np.zeros((1, 1, K_ft))
                returns_2d = returns_outer[:, :, None] + np.zeros((1, 1, K_ft))
                values_2d = values_trajs[:, :, None] + np.zeros((1, 1, K_ft))

                returns_flat = torch.tensor(
                    returns_2d.reshape(N, K_ft), device=self.device, dtype=torch.float32
                )
                values_flat_t = torch.tensor(
                    values_2d.reshape(N, K_ft), device=self.device, dtype=torch.float32
                )
                logprobs_flat_t = torch.tensor(
                    logprobs_trajs, device=self.device, dtype=torch.float32
                )

                # ---------- Q training on env transitions ----------
                actions_flat = actions_d.reshape(N, self.horizon_steps, self.action_dim)
                rewards_flat = (
                    torch.from_numpy(np.ascontiguousarray(reward_trajs.reshape(N)))
                    .float()
                    .to(self.device)
                )
                terminated_flat = (
                    torch.from_numpy(terminated_trajs.reshape(N)).float().to(self.device)
                )

                q_losses = []
                for _ in range(self.q_update_epochs):
                    perm = torch.randperm(N, device=self.device)
                    for start in range(0, N, self.q_minibatch_size):
                        idx = perm[start : start + self.q_minibatch_size]
                        loss_q = self.model.loss_critic_q(
                            obs={k: obs_flat[k][idx] for k in obs_flat},
                            next_obs={k: next_obs_flat[k][idx] for k in next_obs_flat},
                            actions=actions_flat[idx],
                            rewards=rewards_flat[idx],
                            terminated=terminated_flat[idx],
                            gamma=self.gamma,
                        )
                        self.critic_q_optimizer.zero_grad()
                        loss_q.backward()
                        self.critic_q_optimizer.step()
                        q_losses.append(float(loss_q.item()))
                self.model.update_critic_q_target(self.target_ema_rate)
                critic_q_loss_avg = float(np.mean(q_losses)) if q_losses else 0.0

                # ---------- V_inner ----------
                K_total = K_ft + 1
                chains_all = chains_flat_d
                with torch.no_grad():
                    a_term = chains_all[:, -1]
                    v_target = torch.cat(
                        [
                            self.model.compute_q_safe(ob, a_term[st:ed]).view(-1)
                            for st, ed, ob in self._obs_chunks(
                                obs_flat, self.img_encode_chunk
                            )
                        ]
                    )

                v_inner_losses = []
                for _ in range(self.v_inner_update_epochs):
                    perm = torch.randperm(N, device=self.device)
                    for st in range(0, N, self.v_inner_minibatch_size):
                        idx = perm[st : st + self.v_inner_minibatch_size]
                        B = idx.shape[0]
                        # encode the image ONCE per sample, then expand the feature
                        # over the K_total chain positions
                        feat = self.model.critic_v_inner.encode(
                            {k: obs_flat[k][idx] for k in obs_flat}, no_augment=True
                        )
                        feat_exp = feat.unsqueeze(1).expand(
                            B, K_total, feat.shape[-1]
                        ).reshape(B * K_total, feat.shape[-1])
                        x_exp = chains_all[idx].reshape(
                            B * K_total, *chains_all.shape[2:]
                        )
                        k_exp = torch.arange(K_total, device=self.device).repeat(B)
                        tgt_exp = v_target[idx].repeat_interleave(K_total)
                        loss_vi = self.model.loss_critic_v_inner(
                            feat_exp, x_exp, k_exp, tgt_exp
                        )
                        self.v_inner_optimizer.zero_grad()
                        loss_vi.backward()
                        self.v_inner_optimizer.step()
                        v_inner_losses.append(float(loss_vi.item()))
                v_inner_loss_avg = (
                    float(np.mean(v_inner_losses)) if v_inner_losses else 0.0
                )

                # Evaluate V_inner across (N, K_total): encode once per chunk
                with torch.no_grad():
                    V_inner_flat = np.zeros((N, K_total), dtype=np.float32)
                    for st, ed, ob in self._obs_chunks(
                        obs_flat, self.img_encode_chunk
                    ):
                        feat = self.model.critic_v_inner.encode(ob, no_augment=True)
                        for k in range(K_total):
                            k_c = torch.full(
                                (ed - st,), k, dtype=torch.long, device=self.device
                            )
                            V_inner_flat[st:ed, k] = (
                                self.model.v_inner_forward(
                                    feat, chains_all[st:ed, k], k_c
                                )
                                .cpu()
                                .numpy()
                            )
                V_inner_grid = V_inner_flat.reshape(S, E, K_total)
                v_inner_start_mean = float(V_inner_grid[:, :, 0].mean())
                v_inner_final_mean = float(V_inner_grid[:, :, -1].mean())

                # Inner GAE
                A_inner = np.zeros((S, E, K_ft), dtype=np.float32)
                lastgae_in = np.zeros((S, E), dtype=np.float32)
                for k in reversed(range(K_ft)):
                    delta_k = V_inner_grid[:, :, k + 1] - V_inner_grid[:, :, k]
                    A_inner[:, :, k] = lastgae_in = (
                        delta_k + self.v_inner_lambda * lastgae_in
                    )

                sigma_outer = float(advantages_outer.std()) + 1e-8
                sigma_inner = float(A_inner.std()) + 1e-8
                A_inner_scaled = A_inner * (sigma_outer / sigma_inner)
                a_inner_abs_mean = float(np.abs(A_inner).mean())
                a_inner_scaled_mean = float(np.abs(A_inner_scaled).mean())

                advantages_2d = (
                    advantages_outer[:, :, None] + self.v_inner_alpha * A_inner_scaled
                )

                advantages_flat = torch.tensor(
                    advantages_2d.reshape(N, K_ft), device=self.device, dtype=torch.float32
                )

                # ---------- PPO actor + V update ----------
                total_steps = N * K_ft
                clipfracs = []; kls = []; pg_losses = []; v_losses = []
                for update_epoch in range(self.update_epochs):
                    flag_break = False
                    inds_all = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)
                    for ib in range(num_batch):
                        inds_b = inds_all[ib * self.batch_size : (ib + 1) * self.batch_size]
                        batch_inds_b, denoising_inds_b = torch.unravel_index(
                            inds_b, (N, K_ft)
                        )
                        obs_b = {k: obs_flat[k][batch_inds_b] for k in obs_flat}
                        chains_prev_b = chains_flat_d[batch_inds_b, denoising_inds_b]
                        chains_next_b = chains_flat_d[batch_inds_b, denoising_inds_b + 1]
                        returns_b = returns_flat[batch_inds_b, denoising_inds_b]
                        values_b = values_flat_t[batch_inds_b, denoising_inds_b]
                        advantages_b = advantages_flat[batch_inds_b, denoising_inds_b]
                        logprobs_b = logprobs_flat_t[batch_inds_b, denoising_inds_b]

                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_prev_b,
                            chains_next_b,
                            denoising_inds_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs.append(float(clipfrac))
                        kls.append(float(approx_kl))
                        pg_losses.append(float(pg_loss.item()))
                        v_losses.append(float(v_loss.item()))

                        loss.backward()
                        if (ib + 1) % self.grad_accumulate == 0:
                            if self.itr >= self.n_critic_warmup_itr:
                                if self.max_grad_norm is not None:
                                    torch.nn.utils.clip_grad_norm_(
                                        self.model.actor_ft.parameters(),
                                        self.max_grad_norm,
                                    )
                                self.actor_optimizer.step()
                            self.critic_optimizer.step()
                            self.actor_optimizer.zero_grad()
                            self.critic_optimizer.zero_grad()

                            if (
                                self.itr >= self.n_critic_warmup_itr
                                and self.target_kl is not None
                                and approx_kl > self.target_kl
                            ):
                                flag_break = True
                                break
                    if flag_break:
                        log.info(
                            f"actor KL early-stop at epoch {update_epoch}, kl={kls[-1]:.4f}"
                        )
                        break
                pg_loss_avg = float(np.mean(pg_losses)) if pg_losses else 0.0
                critic_loss_avg = float(np.mean(v_losses)) if v_losses else 0.0
                kl_mean = float(np.mean(kls)) if kls else 0.0
                clipfrac_mean = float(np.mean(clipfracs)) if clipfracs else 0.0

                if self.itr >= self.n_critic_warmup_itr:
                    self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
                self.critic_q_lr_scheduler.step()
                self.v_inner_lr_scheduler.step()

            # ---------- Logging ----------
            if eval_mode:
                log.info(
                    f"eval: SR {success_rate:.4f} | avg ep rew {avg_episode_reward:.4f} | "
                    f"avg best {avg_best_reward:.4f}"
                )
            else:
                log.info(
                    f"{self.itr}: step {(self.itr+1)*self.n_envs*self.act_steps*self.n_steps:8d} | "
                    f"pg {pg_loss_avg:+.4f} v {critic_loss_avg:.4f} Q {critic_q_loss_avg:.4f} "
                    f"Vin {v_inner_loss_avg:.4f} | kl {kl_mean:.4f} clip {clipfrac_mean:.3f} | "
                    f"Vin_s={v_inner_start_mean:+.2f} Vin_f={v_inner_final_mean:+.2f} "
                    f"|A_out|={adv_outer_abs_mean:.3f} |A_in|={a_inner_abs_mean:.3f} "
                    f"|A_in_s|={a_inner_scaled_mean:.3f} | "
                    f"reward {avg_episode_reward:.4f} | t:{time.time()-timer_start:.2f}"
                )

            if self.itr % self.save_model_freq == 0:
                self.save_model()

            self.itr += 1
