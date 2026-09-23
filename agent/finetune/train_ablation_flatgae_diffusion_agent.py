"""Flat-GAE ablation: one plain GAE over the whole flattened latent MDP, no alpha, no scale match."""
import os
import math
import time
import einops
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)

from agent.finetune.train_ppo_agent import TrainPPOAgent
from util.scheduler import CosineAnnealingWarmupRestarts


class TrainAblationFlatGAEDiffusionAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # DPPO actor-side knobs (from train_ppo_diffusion_agent)
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)

        # Q optimizer: DPPO's critic optimizer recipe, on the Q ensemble at critic_q_lr.
        critic_q_lr = float(cfg.train.get("critic_q_lr", cfg.train.critic_lr))
        self.critic_q_optimizer = torch.optim.AdamW(
            self.model.critic_q.parameters(),
            lr=critic_q_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_q_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_q_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=critic_q_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.q_update_epochs = int(cfg.train.get("q_update_epochs", 20))
        self.q_minibatch_size = int(cfg.train.get("q_minibatch_size", 1024))
        self.target_ema_rate = float(cfg.train.get("target_ema_rate", 1.0))  # 1.0 = hard copy, no Polyak
        self.reward_offset = float(cfg.train.get("reward_offset", 0.0))

        self.batch_size = int(cfg.train.batch_size)

        # ---------- V_inner: regressed onto Q at the emitted action, then inner GAE ----------
        self._resume_from_itr = int(cfg.train.get("resume_from_itr", 0))
        self.use_v_inner = bool(cfg.train.get("use_v_inner", False))
        self.v_inner_alpha = float(cfg.train.get("v_inner_alpha", 1.0))
        self.v_inner_gamma = float(cfg.train.get("v_inner_gamma", 1.0))
        self.v_inner_lambda = float(cfg.train.get("v_inner_lambda", 0.95))
        self.v_inner_update_epochs = int(cfg.train.get("v_inner_update_epochs", 5))
        self.v_inner_minibatch_size = int(cfg.train.get("v_inner_minibatch_size", 1024))
        # ---------- Flat-GAE latent MDP: bar_gamma in-chain (env gamma at the boundary), one bar_lambda ----------
        self.flat_bar_gamma = float(cfg.train.get("flat_bar_gamma", 1.0))
        self.flat_bar_lambda = float(cfg.train.get("flat_bar_lambda", 0.95))
        # flat_tail_mode: "flat" lets the GAE run into the next chain; "outer" truncates and bootstraps an outer GAE.
        self.flat_tail_mode = str(cfg.train.get("flat_tail_mode", "flat"))
        v_inner_lr = float(cfg.train.get("v_inner_lr", cfg.train.critic_lr))
        if self.use_v_inner:
            assert self.model.critic_v_inner is not None, (
                "use_v_inner=True but model.critic_v_inner is None; "
                "add a critic_v_inner block to the model config."
            )
            self.v_inner_optimizer = torch.optim.AdamW(
                self.model.critic_v_inner.parameters(),
                lr=v_inner_lr,
                weight_decay=cfg.train.critic_weight_decay,
            )
            self.v_inner_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.v_inner_optimizer,
                first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=v_inner_lr,
                min_lr=cfg.train.critic_lr_scheduler.min_lr,
                warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
                gamma=1.0,
            )

    def run(self):
        timer_start = time.time()
        # Resume from a saved checkpoint if resume_from_itr is set (else start at 0).
        resume_itr = int(getattr(self, "_resume_from_itr", 0))
        if resume_itr > 0:
            self.load(resume_itr)
            self.itr = resume_itr + 1
            log.info(f"Resumed from iter {resume_itr}, starting at iter {self.itr}")
        else:
            self.itr = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
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
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv
            last_itr_eval = eval_mode

            obs_trajs = {
                "state": np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim)
                )
            }
            next_obs_trajs = {
                "state": np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim)
                )
            }
            K_ft = self.model.ft_denoising_steps
            chains_trajs = np.zeros(
                (self.n_steps, self.n_envs, K_ft + 1, self.horizon_steps, self.action_dim)
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            action_trajs = np.zeros(
                (self.n_steps, self.n_envs, self.horizon_steps, self.action_dim)
            )

            # ---------- Rollout ----------
            for step in range(self.n_steps):
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                    }
                    samples = self.model(cond=cond, deterministic=eval_mode, return_chain=True)
                    output_venv = samples.trajectories.cpu().numpy()
                    chains_venv = samples.chains.cpu().numpy()
                action_venv = output_venv[:, :self.act_steps]

                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                obs_trajs["state"][step] = prev_obs_venv["state"]
                chains_trajs[step] = chains_venv
                action_trajs[step] = output_venv
                reward_trajs[step] = reward_venv + self.reward_offset
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv
                # robomimic puts the pre-reset obs in info["final_obs"]; furniture returns it directly.
                robomimic_info = isinstance(info_venv, (list, tuple))
                for i in range(self.n_envs):
                    if robomimic_info and truncated_venv[i] and "final_obs" in info_venv[i]:
                        next_obs_trajs["state"][step, i] = info_venv[i]["final_obs"]["state"]
                    else:
                        next_obs_trajs["state"][step, i] = obs_venv["state"][i]
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
                    reward_trajs[s_:e_ + 1, env_ind]
                    for env_ind, s_, e_ in episodes_start_end
                ]
                episode_reward = np.array([np.sum(r) for r in rew_split])
                if self.furniture_sparse_reward:
                    # furniture reward is sparse-terminal, so use the episode sum, not the act_steps mean.
                    episode_best_reward = episode_reward
                else:
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

            # ---------- Update ----------
            critic_loss_avg = 0.0
            critic_q_loss_avg = 0.0
            pg_loss_avg = 0.0
            adv_outer_abs_mean = 0.0
            kl_mean = 0.0
            clipfrac_mean = 0.0

            if not eval_mode:
                S, E = self.n_steps, self.n_envs
                N = S * E

                with torch.no_grad():
                    obs_state_d = torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                    next_obs_state_d = torch.from_numpy(next_obs_trajs["state"]).float().to(self.device)
                    chains_d = torch.from_numpy(chains_trajs).float().to(self.device)
                    actions_d = chains_d[:, :, -1]                  # (S, E, H, A) executed actions

                    # V (state-only) per env step
                    values_flat = self.model.critic(
                        {"state": obs_state_d.reshape(N, *obs_state_d.shape[2:])}
                    ).view(S, E)
                    values_trajs = values_flat.cpu().numpy()

                    # Bootstrap V at the post-rollout obs
                    obs_venv_ts = {
                        "state": torch.from_numpy(obs_venv["state"]).float().to(self.device)
                    }
                    boot_v = self.model.critic(obs_venv_ts).view(-1).cpu().numpy()

                    # Per-K log-probs
                    logprobs_trajs = self.model.get_logprobs(
                        {"state": obs_state_d.reshape(N, *obs_state_d.shape[2:])},
                        chains_d.reshape(N, K_ft + 1, self.horizon_steps, self.action_dim),
                    ).cpu().numpy().reshape(N, K_ft, self.horizon_steps, self.action_dim)

                # ---------- Running reward scaling ----------
                if self.reward_scale_running:
                    reward_trajs_t = self.running_reward_scaler(
                        reward=reward_trajs.T, first=firsts_trajs[:-1].T
                    )
                    reward_trajs_scaled = reward_trajs_t.T
                else:
                    reward_trajs_scaled = reward_trajs

                # ---------- Outer GAE (env-step level) ----------
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

                # ---------- 2D per-K advantages: A_outer broadcast; A_inner added below ----------
                advantages_2d = advantages_outer[:, :, None] + np.zeros((1, 1, K_ft))  # (S, E, K_ft)
                returns_2d = (
                    returns_outer[:, :, None] + np.zeros((1, 1, K_ft))            # returns broadcast (used for V loss)
                )
                values_2d = values_trajs[:, :, None] + np.zeros((1, 1, K_ft))

                # ---------- Flatten to (batch_idx, denoising_idx) over S*E*K_ft for the actor loop ----------
                obs_flat_d = obs_state_d.reshape(N, *obs_state_d.shape[2:])         # (N, To, Do)
                chains_flat_d = chains_d.reshape(N, K_ft + 1, self.horizon_steps, self.action_dim)
                advantages_flat = torch.tensor(
                    advantages_2d.reshape(N, K_ft), device=self.device, dtype=torch.float32
                )                                                                   # (N, K_ft)
                returns_flat = torch.tensor(
                    returns_2d.reshape(N, K_ft), device=self.device, dtype=torch.float32
                )                                                                   # (N, K_ft)
                values_flat_t = torch.tensor(
                    values_2d.reshape(N, K_ft), device=self.device, dtype=torch.float32
                )                                                                   # (N, K_ft)
                logprobs_flat_t = torch.tensor(
                    logprobs_trajs, device=self.device, dtype=torch.float32
                )                                                                   # (N, K_ft, H, A)

                # ---------- Q training on env transitions ----------
                obs_flat_for_q = obs_state_d.reshape(N, *obs_state_d.shape[2:])
                next_obs_flat_for_q = next_obs_state_d.reshape(N, *next_obs_state_d.shape[2:])
                actions_flat_for_q = actions_d.reshape(N, self.horizon_steps, self.action_dim)
                # Q trains on the raw env reward, so its target stays in return units.
                reward_for_q = reward_trajs
                rewards_flat_for_q = torch.from_numpy(
                    np.ascontiguousarray(reward_for_q.reshape(N))
                ).float().to(self.device)
                terminated_flat_for_q = torch.from_numpy(
                    terminated_trajs.reshape(N)
                ).float().to(self.device)

                q_losses = []
                for _ in range(self.q_update_epochs):
                    perm = torch.randperm(N, device=self.device)
                    for start in range(0, N, self.q_minibatch_size):
                        idx = perm[start:start + self.q_minibatch_size]
                        loss_q = self.model.loss_critic_q(
                            obs={"state": obs_flat_for_q[idx]},
                            next_obs={"state": next_obs_flat_for_q[idx]},
                            actions=actions_flat_for_q[idx],
                            rewards=rewards_flat_for_q[idx],
                            terminated=terminated_flat_for_q[idx],
                            gamma=self.gamma,
                        )
                        self.critic_q_optimizer.zero_grad()
                        loss_q.backward()
                        self.critic_q_optimizer.step()
                        q_losses.append(float(loss_q.item()))
                # refresh the Q target; at target_ema_rate 1.0 this is a hard copy
                self.model.update_critic_q_target(self.target_ema_rate)
                critic_q_loss_avg = float(np.mean(q_losses)) if q_losses else 0.0

                # ---------- V_inner: one target per chain, Q_target(s, chain[K_ft]), shared by all k ----------
                v_inner_loss_avg = 0.0
                a_inner_abs_mean = 0.0
                a_inner_scaled_mean = 0.0
                v_inner_start_mean = 0.0
                v_inner_final_mean = 0.0
                if self.use_v_inner:
                    with torch.no_grad():
                        cond_term = {"state": obs_flat_for_q}
                        a_term = chains_d[:, :, -1].reshape(N, self.horizon_steps, self.action_dim)
                        v_target = self.model.compute_q_safe(cond_term, a_term).view(N)
                    # Stack (s, x_k, k, target) for all chain positions
                    K_total = K_ft + 1
                    cond_all = obs_state_d.reshape(N, *obs_state_d.shape[2:])
                    chains_all = chains_d.reshape(N, K_total, self.horizon_steps, self.action_dim)
                    v_inner_losses = []
                    # Expand each sample over all K_total positions, so every k is covered each epoch.
                    for _ in range(self.v_inner_update_epochs):
                        perm = torch.randperm(N, device=self.device)
                        for st in range(0, N, self.v_inner_minibatch_size):
                            idx = perm[st:st + self.v_inner_minibatch_size]
                            B = idx.shape[0]
                            cond_exp = cond_all[idx].unsqueeze(1).expand(
                                B, K_total, *cond_all.shape[1:]
                            ).reshape(B * K_total, *cond_all.shape[1:])
                            x_exp = chains_all[idx].reshape(
                                B * K_total, *chains_all.shape[2:]
                            )
                            k_exp = torch.arange(K_total, device=self.device).repeat(B)
                            tgt_exp = v_target[idx].repeat_interleave(K_total)
                            cond_b = {"state": cond_exp}
                            loss_vi = self.model.loss_critic_v_inner(cond_b, x_exp, k_exp, tgt_exp)
                            self.v_inner_optimizer.zero_grad()
                            loss_vi.backward()
                            self.v_inner_optimizer.step()
                            v_inner_losses.append(float(loss_vi.item()))
                    v_inner_loss_avg = float(np.mean(v_inner_losses)) if v_inner_losses else 0.0
                    # Evaluate V_inner across (S, E, K_ft+1) for inner GAE
                    with torch.no_grad():
                        V_inner_grid = np.zeros((S, E, K_total), dtype=np.float32)
                        chunk = self.logprob_batch_size
                        for k in range(K_total):
                            for st in range(0, N, chunk):
                                ed = min(st + chunk, N)
                                cond_c = {"state": cond_all[st:ed]}
                                x_c = chains_all[st:ed, k]
                                k_c = torch.full((ed - st,), k, dtype=torch.long, device=self.device)
                                vi = self.model.v_inner_forward(cond_c, x_c, k_c).cpu().numpy()
                                for off in range(ed - st):
                                    s_g = (st + off) // E
                                    e_g = (st + off) % E
                                    V_inner_grid[s_g, e_g, k] = vi[off]
                    v_inner_start_mean = float(V_inner_grid[:, :, 0].mean())
                    v_inner_final_mean = float(V_inner_grid[:, :, -1].mean())

                    # ---------- Flat GAE: r_t only at the boundary micro-step, unbroken across chains, no alpha ----------
                    Vbar = V_inner_grid            # (S, E, K_ft+1); positions 0..K_ft-1 are the chain states
                    bg, bl, g = self.flat_bar_gamma, self.flat_bar_lambda, self.gamma
                    # Terminal bootstrap on the V_inner/Q scale: V_bar(s_S, noise, 0) == E_a[Q(s_S, a)].
                    with torch.no_grad():
                        a_boot_s = self.model(cond=obs_venv_ts, deterministic=False).trajectories
                        boot_vbar = (
                            self.model.compute_q_safe(obs_venv_ts, a_boot_s).view(-1).cpu().numpy()
                        )
                    reward_bnd = reward_for_q       # (S, E) env reward on the V_inner/Q scale

                    # flat_tail_mode="outer": truncate at the boundary and bootstrap a separate outer GAE on the Q scale.
                    A_outer_q = None
                    if self.flat_tail_mode == "outer":
                        V_noise = Vbar[:, :, 0]                       # (S, E), == V(s) on Q scale
                        lam_outer = bl ** K_ft
                        A_outer_q = np.zeros((S, E), dtype=np.float32)
                        lastgae_o = np.zeros(E, dtype=np.float32)
                        for t in reversed(range(S)):
                            nonterm_o = 1.0 - terminated_trajs[t]
                            nextv = V_noise[t + 1] if t < S - 1 else boot_vbar
                            delta_o = reward_bnd[t] + g * nextv * nonterm_o - V_noise[t]
                            A_outer_q[t] = lastgae_o = (
                                delta_o + g * lam_outer * nonterm_o * lastgae_o
                            )

                    A_flat = np.zeros((S, E, K_ft), dtype=np.float32)
                    next_chain_adv = np.zeros(E, dtype=np.float32)   # A_hat at (t+1, k=0)
                    next_chain_v0 = boot_vbar.copy()                 # V_bar at (t+1, k=0)
                    for t in reversed(range(S)):
                        nonterm = 1.0 - terminated_trajs[t]          # (E,)
                        # boundary micro-step k=K_ft-1: env reward r_t, bootstrap next chain (gamma)
                        Vb_last = Vbar[t, :, K_ft - 1]
                        delta_b = reward_bnd[t] + g * next_chain_v0 * nonterm - Vb_last
                        # tail: flat-recursive (default) OR outer-GAE substitution (writeup)
                        if self.flat_tail_mode == "outer":
                            tail_adv = (
                                A_outer_q[t + 1] if t < S - 1
                                else np.zeros(E, dtype=np.float32)
                            )
                        else:
                            tail_adv = next_chain_adv
                        lastgae = delta_b + g * bl * nonterm * tail_adv
                        A_flat[t, :, K_ft - 1] = lastgae
                        # within-chain micro-steps k=K_ft-2..0: no reward, discount bar_gamma
                        for k in reversed(range(K_ft - 1)):
                            delta_k = bg * Vbar[t, :, k + 1] - Vbar[t, :, k]
                            lastgae = delta_k + bg * bl * lastgae
                            A_flat[t, :, k] = lastgae
                        next_chain_adv = A_flat[t, :, 0]
                        next_chain_v0 = Vbar[t, :, 0]
                    a_inner_abs_mean = float(np.abs(A_flat).mean())
                    a_inner_scaled_mean = a_inner_abs_mean
                    advantages_2d = A_flat                             # (S, E, K_ft)
                    advantages_flat = torch.tensor(
                        advantages_2d.reshape(N, K_ft), device=self.device, dtype=torch.float32
                    )

                # ---------- PPO actor + V: run every iter so V_outer trains during actor warmup ----------
                if True:
                    total_steps = N * K_ft
                    clipfracs = []; kls = []; pg_losses = []; v_losses = []
                    for update_epoch in range(self.update_epochs):
                        flag_break = False
                        inds_all = torch.randperm(total_steps, device=self.device)
                        num_batch = max(1, total_steps // self.batch_size)
                        for ib in range(num_batch):
                            inds_b = inds_all[ib * self.batch_size:(ib + 1) * self.batch_size]
                            batch_inds_b, denoising_inds_b = torch.unravel_index(
                                inds_b, (N, K_ft)
                            )
                            obs_b = {"state": obs_flat_d[batch_inds_b]}
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

                            self.actor_optimizer.zero_grad()
                            self.critic_optimizer.zero_grad()
                            loss.backward()
                            if self.itr >= self.n_critic_warmup_itr:
                                self.actor_optimizer.step()   # actor gated by warmup
                            self.critic_optimizer.step()      # V_outer trains every iter

                            if self.itr >= self.n_critic_warmup_itr and self.target_kl is not None and approx_kl > self.target_kl:
                                flag_break = True
                                break
                        if flag_break:
                            log.info(f"actor KL early-stop at epoch {update_epoch}, kl={kls[-1]:.4f}")
                            break
                    pg_loss_avg = float(np.mean(pg_losses)) if pg_losses else 0.0
                    critic_loss_avg = float(np.mean(v_losses)) if v_losses else 0.0
                    kl_mean = float(np.mean(kls)) if kls else 0.0
                    clipfrac_mean = float(np.mean(clipfracs)) if clipfracs else 0.0

                # LR step
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
                self.critic_q_lr_scheduler.step()
                if self.use_v_inner:
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
                    f"pg {pg_loss_avg:+.4f} v {critic_loss_avg:.4f} Q {critic_q_loss_avg:.4f} Vin {v_inner_loss_avg:.4f} | "
                    f"kl {kl_mean:.4f} clip {clipfrac_mean:.3f} | "
                    f"Vin_s={v_inner_start_mean:+.2f} Vin_f={v_inner_final_mean:+.2f} "
                    f"|A_out|={adv_outer_abs_mean:.3f} |A_in|={a_inner_abs_mean:.3f} |A_in_s|={a_inner_scaled_mean:.3f} | "
                    f"reward {avg_episode_reward:.4f} | t:{time.time()-timer_start:.2f}"
                )

            # ---------- Save model ----------
            if self.itr % self.save_model_freq == 0:
                self.save_model()

            self.itr += 1

