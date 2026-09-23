"""DIA: DPPO plus an inner-value advantage over the denoising chain.

Standard DPPO actor/V training path (uses PPODiffusion.loss for the actor, so it
inherits the time-varying clip, joint optimization, value loss and entropy).
Adds, on top of that:

  - Q critic: an n-head CriticObsAct ensemble trained on env transitions. n_heads
    comes from the config and is 2 in every paper config except kitchen, which
    uses 10. Its target network is refreshed by `target_ema_rate`, which is 1.0 in
    every paper config, so the refresh is a hard copy and not a Polyak average.
    The heads are aggregated by `q_aggregation` ("mean" everywhere). No CQL term.
  - Regression target for V_inner: the aggregated target-Q evaluated at the action
    the denoising chain actually emits, Q(s, a_0).
  - V_inner: CriticObsInnerState, V_in(s, x_k, k), regressed onto that Q. It is the
    value of a partially denoised action, so it varies along the chain.
  - Inner advantage: an inner GAE over V_inner along k, with zero inner reward and
    no within-chain discount, only the trace decay v_inner_lambda. Its scale is
    then matched to the outer advantage globally, A_in_scaled = A_in *
    (sigma_out / sigma_in), which is what makes alpha a meaningful mixing weight
    rather than a scale knob.
  - Combined 2D advantage: adv[s, e, k] = A_outer[s, e] + alpha * A_in_scaled[s, e, k],
    with A_outer from the standard outer GAE.

The DIA_DUMP_ADV env var dumps the advantage arrays for one iteration and exits.
The inner-noise ablation lives in train_ablation_dia_noise_agent.py.

Vanilla DPPO code path only.
"""
import os
import math
import time
import einops
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)

# Fallbacks used only when the config does not set these keys. Every config that
# ships in cfg/ sets them explicitly, so these values change no shipped result;
# they exist so the knobs stay settable and get recorded like every other one.
# The sibling flat-GAE agent reads q_update_epochs the same way.
Q_UPDATE_EPOCHS = 20
V_INNER_MINIBATCH_SIZE = 1024

from agent.finetune.train_ppo_agent import TrainPPOAgent
from util.scheduler import CosineAnnealingWarmupRestarts


class TrainPPODIADiffusionAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # DPPO actor-side knobs (from train_ppo_diffusion_agent)
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)

        # Q critic optimizer. Built exactly like DPPO's outer value critic
        # optimizer in train_ppo_agent: AdamW with the same weight decay, and a
        # cosine schedule with warmup restarts on the same cycle/min-lr/warmup
        # settings. The only difference is that it optimizes the Q ensemble and
        # uses critic_q_lr, which falls back to critic_lr when unset.
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
        self.q_update_epochs = int(cfg.train.get("q_update_epochs", Q_UPDATE_EPOCHS))
        self.q_minibatch_size = int(cfg.train.get("q_minibatch_size", 1024))
        self.target_ema_rate = float(cfg.train.get("target_ema_rate", 1.0))  # 1.0 = hard copy, no Polyak
        self.q_scale_reward_factor = float(cfg.train.get("q_scale_reward_factor", 1.0))

        self.batch_size = int(cfg.train.batch_size)

        # ---------- V_inner: the inner-MDP value ----------
        # Train critic_v_inner(s, x_k, k) by regressing it onto the target-Q at the
        # chain's emitted action, Q_target(s, chain[K_ft]); then form A_inner by an
        # inner GAE along k (zero inner reward, lambda = v_inner_lambda) and add it to
        # A_outer after matching its std, so alpha is a mixing weight not a scale knob.
        self.v_inner_alpha = float(cfg.train.get("v_inner_alpha", 1.0))
        self.v_inner_lambda = float(cfg.train.get("v_inner_lambda", 0.95))
        self.v_inner_update_epochs = int(cfg.train.get("v_inner_update_epochs", 5))
        self.v_inner_minibatch_size = int(
            cfg.train.get("v_inner_minibatch_size", V_INNER_MINIBATCH_SIZE)
        )
        if cfg.train.get("inner_noise_ablation", False):
            raise ValueError(
                "inner_noise_ablation moved to "
                "agent.finetune.train_ablation_dia_noise_agent."
                "TrainAblationDiaNoiseAgent; point _target_ at it instead of "
                "setting this flag, which this agent no longer honours"
            )
        v_inner_lr = float(cfg.train.get("v_inner_lr", cfg.train.critic_lr))
        assert self.model.critic_v_inner is not None, (
            "DIA requires model.critic_v_inner; add a critic_v_inner block to the "
            "model config, or use the DPPO agent for no inner advantage."
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

    def scale_match_inner(self, A_inner, advantages_outer):
        """Rescale the inner advantage to the outer advantage's standard deviation.

        Global over the batch, not per denoising step k, so alpha means the same
        thing at every k."""
        sigma_outer = float(advantages_outer.std()) + 1e-8
        sigma_inner = float(A_inner.std()) + 1e-8
        return A_inner * (sigma_outer / sigma_inner)

    def run(self):
        timer_start = time.time()
        # Resume from a saved checkpoint if resume_from_itr is set (else start at 0).
        resume_itr = int(self.cfg.train.get("resume_from_itr", 0))
        if resume_itr > 0:
            self.load(resume_itr)
            self.itr = resume_itr + 1
            log.info(f"Resumed from iter {resume_itr}, starting at iter {self.itr}")
        else:
            self.itr = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        # Bound before the loop: with reset_at_iteration=False (kitchen) a resumed run
        # would otherwise reach the rollout with prev_obs_venv unassigned.
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
                or prev_obs_venv is None   # first pass of a resumed run: nothing to carry over
            ):
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
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv
                # robomimic: info_venv is a per-env list of dicts carrying "final_obs" on
                # truncation (the multi_step wrapper resets within the step). furniture: the env
                # is natively vectorized, info_venv is a single batched dict, and with
                # reset_within_step=False obs_venv is already the true next obs (no final_obs).
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
                    # furniture: reward occurs in a single env step (sparse terminal), so the
                    # episode sum IS the best reward; do NOT divide by act_steps (that's a
                    # robomimic-only normalization and would zero out the success rate).
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

                # ---------- 2D per-K advantages: base = A_outer broadcast ----------
                # A_inner (inner GAE over V_inner) is added in the V_inner block below.
                advantages_2d = advantages_outer[:, :, None] + np.zeros((1, 1, K_ft))  # (S, E, K_ft)
                returns_2d = (
                    returns_outer[:, :, None] + np.zeros((1, 1, K_ft))            # returns broadcast (used for V loss)
                )
                values_2d = values_trajs[:, :, None] + np.zeros((1, 1, K_ft))

                # ---------- Convert to flat tensors for actor PPO loop ----------
                # Sampling indexes: (batch_idx, denoising_idx) over total = S*E*K_ft
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
                # Q is trained on the raw environment reward, not the running-scaled
                # reward the outer GAE uses, so the Q target and V_inner stay in return units.
                rewards_flat_for_q = torch.from_numpy(
                    np.ascontiguousarray(reward_trajs.reshape(N))
                ).float().to(self.device) * self.q_scale_reward_factor
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

                # ---------- V_inner: regress to the target Q at the emitted action x_K ----------
                # Inner MDP: r_inner_per_step = 0, terminal value = Q_target(s, chain[K_ft]).
                # Target is the SAME scalar for all k on a chain — but inputs vary, so V_inner
                # ends up reflecting "how much chain[k] reveals about Q(s, terminal)" per k.
                v_inner_loss_avg = 0.0
                a_inner_abs_mean = 0.0
                a_inner_scaled_mean = 0.0
                v_inner_start_mean = 0.0
                v_inner_final_mean = 0.0
                with torch.no_grad():
                    cond_term = {"state": obs_flat_for_q}
                    a_term = chains_d[:, :, -1].reshape(N, self.horizon_steps, self.action_dim)
                    v_target = self.model.compute_q_safe(cond_term, a_term).view(N)
                # Stack (s, x_k, k, target) for all chain positions
                K_total = K_ft + 1
                cond_all = obs_state_d.reshape(N, *obs_state_d.shape[2:])
                chains_all = chains_d.reshape(N, K_total, self.horizon_steps, self.action_dim)
                v_inner_losses = []
                # Expand each sample to ALL K_total chain positions per minibatch.
                # B base samples → B*K_total (s, x_k, k, target) triples each minibatch.
                # Guarantees full per-k coverage every epoch (vs random-k sampling).
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

                # Inner GAE: δ_k = V[k+1] - V[k]; A[k] = δ_k + λ_in A[k+1].
                # No within-chain discount: the recursion below has no γ_in factor.
                A_inner = np.zeros((S, E, K_ft), dtype=np.float32)
                lastgae_in = np.zeros((S, E), dtype=np.float32)
                for k in reversed(range(K_ft)):
                    delta_k = V_inner_grid[:, :, k + 1] - V_inner_grid[:, :, k]
                    A_inner[:, :, k] = lastgae_in = (
                        delta_k + self.v_inner_lambda * lastgae_in
                    )

                A_inner_scaled = self.scale_match_inner(A_inner, advantages_outer)
                a_inner_abs_mean = float(np.abs(A_inner).mean())
                a_inner_scaled_mean = float(np.abs(A_inner_scaled).mean())


                if os.environ.get("DIA_DUMP_ADV"):
                    _vt = v_target.detach().cpu().numpy() if hasattr(v_target, "detach") else np.asarray(v_target)
                    np.savez(
                        os.environ["DIA_DUMP_ADV"],
                        A_out=advantages_outer, A_inner_scaled=A_inner_scaled,
                        reward=np.asarray(reward_trajs), reward_scaled=np.asarray(reward_trajs_scaled),
                        values=values_trajs, terminated=np.asarray(terminated_trajs),
                        firsts=np.asarray(firsts_trajs), boot_v=boot_v,
                        v_target=_vt.reshape(S, E), gamma=self.gamma, gae_lambda=self.gae_lambda,
                        alpha=self.v_inner_alpha, S=S, E=E, K=K_ft,
                    )
                    print("DUMPED_ADV ->", os.environ["DIA_DUMP_ADV"], "S,E,K=", S, E, K_ft, flush=True)
                    os._exit(0)

                # Override advantages_2d / advantages_flat
                advantages_2d = (
                    advantages_outer[:, :, None] + self.v_inner_alpha * A_inner_scaled
                )
                advantages_flat = torch.tensor(
                    advantages_2d.reshape(N, K_ft), device=self.device, dtype=torch.float32
                )

                # ---------- PPO actor + V update via PPODiffusion.loss ----------
                # Run EVERY iter so V_outer (critic) warms up while the actor is frozen;
                # the actor optimizer step is gated by warmup below (DPPO parity). Without
                # this, V_outer got zero training during warmup -> untrained value baseline
                # at activation -> garbage A_outer -> the first actor step crashed reward.
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

                # LR step. The actor schedule is gated on the critic warmup exactly as
                # in train_ppo_diffusion_agent: during warmup the actor is frozen, so
                # advancing its cosine schedule would leave the two agents on different
                # actor LRs for the rest of the run.
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

