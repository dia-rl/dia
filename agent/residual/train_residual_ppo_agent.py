"""ResiP fine-tuning: PPO on the residual, DPPO's diffusion base frozen."""

import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import logging
import wandb

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent


def cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5
):
    """diffusers' `get_scheduler(name="cosine")`, which is what ResiP calls.

    Reimplemented so this baseline adds no dependency, and so it cannot be confused
    with the repository's `CosineAnnealingWarmupRestarts`, a different schedule.
    """

    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TrainResidualPPOAgent(TrainPPODiffusionAgent):

    def __init__(self, cfg):
        # DPPO's full setup: env, model, gamma, reward scaler, warmup, epochs, checkpointing.
        super().__init__(cfg)

        rp = cfg.train.resip
        self.residual_policy = self.model.residual_policy

        # The residual acts once per env step, so the wrapper runs one step per policy call.
        if "specific" in cfg.env and cfg.env.get("env_type") == "furniture":
            per_call, where = cfg.env.specific.act_steps, "env.specific.act_steps"
        else:
            per_call = cfg.env.wrappers.multi_step.n_action_steps
            where = "env.wrappers.multi_step.n_action_steps"
        assert per_call == 1, f"set {where}=1; the residual is closed loop (got {per_call})"
        assert (
            self.n_steps % self.act_steps == 0
        ), "train.n_steps counts env steps and must be a whole number of act_steps chunks"
        self.n_chunks = self.n_steps // self.act_steps

        # ---- ResiP's algorithm hyperparameters ----
        self.rp_discount = rp.discount
        self.rp_gae_lambda = rp.gae_lambda
        self.rp_norm_adv = rp.norm_adv
        self.rp_clip_coef = rp.clip_coef
        self.rp_clip_vloss = rp.clip_vloss
        self.rp_ent_coef = rp.ent_coef
        self.rp_vf_coef = rp.vf_coef
        self.rp_target_kl = rp.target_kl
        self.rp_update_epochs = rp.update_epochs
        self.rp_num_minibatches = rp.num_minibatches
        self.rp_l1 = rp.residual_l1
        self.rp_l2 = rp.residual_l2
        self.rp_train_only_value = rp.n_iterations_train_only_value
        self.rp_max_grad_norm = rp.max_grad_norm

        self.rollout_batch_size = self.n_steps * self.n_envs
        self.minibatch_size = self.rollout_batch_size // self.rp_num_minibatches

        # ResiP's optimizers replace the base class's actor_ft/critic ones, which this method never trains.
        self.actor_optimizer = torch.optim.AdamW(
            self.residual_policy.actor_parameters,
            lr=rp.learning_rate_actor,
            betas=tuple(rp.optimizer_betas_actor),
            eps=1e-5,
            weight_decay=1e-6,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.residual_policy.critic_parameters,
            lr=rp.learning_rate_critic,
            eps=1e-5,
            weight_decay=1e-6,
        )
        # Keep upstream's schedule formula, which leaves the cosine leg nearly flat over our shorter run.
        self.lr_num_training_steps = int(rp.total_timesteps // self.rollout_batch_size)
        self.actor_lr_scheduler = cosine_schedule_with_warmup(
            self.actor_optimizer, rp.lr_scheduler.actor_warmup_steps, self.lr_num_training_steps
        )
        self.critic_lr_scheduler = cosine_schedule_with_warmup(
            self.critic_optimizer, rp.lr_scheduler.critic_warmup_steps, self.lr_num_training_steps
        )

        log.info(
            "ResiP: %d env steps x %d envs = %d transitions/iter (%d chunks of %d), "
            "%d minibatch(es) of %d, %d epochs, critic warmup %d iters (DPPO's)",
            self.n_steps, self.n_envs, self.rollout_batch_size, self.n_chunks,
            self.act_steps, self.rp_num_minibatches, self.minibatch_size,
            self.rp_update_epochs, self.n_critic_warmup_itr,
        )

    # ------------------------------------------------------------------ helpers
    def _to_chunks(self, per_step, how):
        """(n_steps, n_envs) per environment step -> (n_chunks, n_envs) per chunk.

        Reproduces the buffer DPPO would have collected from the same rollout: its
        multi_step wrapper sums reward over the chunk and reports the chunk's done as
        the max over its steps. Feeding this to DPPO's own statistics and reward
        scaling makes those quantities identical across the arms.
        """
        x = per_step.reshape(self.n_chunks, self.act_steps, self.n_envs)
        return x.sum(axis=1) if how == "sum" else x.max(axis=1)

    # --------------------------------------------------------------------- run
    def run(self):
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        resume_itr = int(self.cfg.train.get("resume_from_itr", 0))
        if resume_itr > 0:
            self.load(resume_itr)
            self.itr = resume_itr + 1
            log.info(f"Resumed ResiP from iter {resume_itr}, starting at iter {self.itr}")
        prev_obs_venv = None

        while self.itr < self.n_train_itr:
            # ---- identical to TrainPPODiffusionAgent.run() ----
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if (
                self.reset_at_iteration
                or eval_mode
                or last_itr_eval
                or prev_obs_venv is None
            ):
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                self.model.reset_chunks(self.n_envs)
                firsts_trajs[0] = 1
                done_venv = np.zeros(self.n_envs, dtype=bool)
            else:
                firsts_trajs[0] = done_venv
            # ---- end identical block ----

            obs_dim_res = self.residual_policy.obs_dim
            res_obs_trajs = torch.zeros((self.n_steps, self.n_envs, obs_dim_res))
            res_act_trajs = torch.zeros((self.n_steps, self.n_envs, self.action_dim))
            logprob_trajs = torch.zeros((self.n_steps, self.n_envs))
            value_trajs = torch.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))

            done_venv = np.asarray(done_venv, dtype=bool).reshape(-1)
            for step in range(self.n_steps):
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                }
                with torch.no_grad():
                    # base sampled by DPPO's code; deterministic at eval as DPPO does
                    base_a = self.model.base_action(
                        cond, deterministic=eval_mode, force=torch.from_numpy(done_venv)
                    )
                    res_obs = self.model.residual_obs(cond["state"], base_a)
                    samp, logprob, _, value, mean = (
                        self.residual_policy.get_action_and_value(res_obs)
                    )
                res_a = mean if eval_mode else samp
                action_venv = self.model.compose(base_a, res_a).cpu().numpy()[:, None, :]

                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv

                res_obs_trajs[step] = res_obs.cpu()
                res_act_trajs[step] = res_a.cpu()
                logprob_trajs[step] = logprob.cpu()
                value_trajs[step] = value.flatten().cpu()
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                prev_obs_venv = obs_venv
                cnt_train_step += self.n_envs if not eval_mode else 0

            # ---- episode statistics: DPPO's block, on the chunk-aggregated view so it is the same quantity ----
            chunk_rewards = self._to_chunks(reward_trajs, "sum")
            chunk_firsts = np.concatenate(
                [firsts_trajs[:-1].reshape(self.n_chunks, self.act_steps, self.n_envs).max(axis=1),
                 firsts_trajs[-1:][None].reshape(1, self.n_envs)],
                axis=0,
            )
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(chunk_firsts[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start, end = env_steps[i], env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    chunk_rewards[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array([np.sum(r) for r in reward_trajs_split])
                if self.furniture_sparse_reward:
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [np.max(r) / self.act_steps for r in reward_trajs_split]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")
            # ---- end verbatim block ----

            if not eval_mode:
                # ---- reward scaling: DPPO's scaler fit on chunk-level rewards, divisor applied per step ----
                if self.reward_scale_running:
                    self.running_reward_scaler(
                        reward=chunk_rewards.T, first=chunk_firsts[:-1].T
                    )
                    div = np.sqrt(
                        self.running_reward_scaler.ret_rms.var
                        + self.running_reward_scaler.epsilon
                    )
                    reward_scaled = np.clip(
                        reward_trajs / div,
                        -self.running_reward_scaler.cliprew,
                        self.running_reward_scaler.cliprew,
                    )
                else:
                    reward_scaled = reward_trajs
                reward_scaled = reward_scaled * self.reward_scale_const

                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                    }
                    base_a = self.model.base_action(
                        cond, force=torch.from_numpy(done_venv), advance=False
                    )
                    next_value = (
                        self.residual_policy.get_value(
                            self.model.residual_obs(cond["state"], base_a)
                        ).reshape(1, -1).cpu().numpy()
                    )

                    # ResiP's GAE on the env-step MDP, masking on `terminated` only, as DPPO does.
                    values = value_trajs.numpy()
                    advantages = np.zeros_like(reward_scaled)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        nextvalues = next_value if t == self.n_steps - 1 else values[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        delta = (
                            reward_scaled[t]
                            + self.rp_discount * nextvalues * nonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + self.rp_discount * self.rp_gae_lambda * nonterminal * lastgaelam
                        )
                    returns = advantages + values

                b_obs = res_obs_trajs.reshape(-1, obs_dim_res)
                b_act = res_act_trajs.reshape(-1, self.action_dim)
                b_logp = logprob_trajs.reshape(-1)
                b_val = value_trajs.reshape(-1)
                b_adv = torch.from_numpy(advantages.reshape(-1)).float()
                b_ret = torch.from_numpy(returns.reshape(-1)).float()

                # ---- ResiP's PPO update ----
                inds = np.arange(self.rollout_batch_size)
                clipfracs = []
                for epoch in range(self.rp_update_epochs):
                    early_stop = False
                    np.random.shuffle(inds)
                    for start in range(0, self.rollout_batch_size, self.minibatch_size):
                        mb = inds[start : start + self.minibatch_size]
                        mb_obs = b_obs[mb].to(self.device)
                        mb_act = b_act[mb].to(self.device)
                        mb_logp = b_logp[mb].to(self.device)
                        mb_adv = b_adv[mb].to(self.device)
                        mb_ret = b_ret[mb].to(self.device)
                        mb_val = b_val[mb].to(self.device)

                        _, newlogprob, entropy, newvalue, action_mean = (
                            self.residual_policy.get_action_and_value(mb_obs, mb_act)
                        )
                        logratio = newlogprob - mb_logp
                        ratio = logratio.exp()
                        with torch.no_grad():
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [
                                ((ratio - 1.0).abs() > self.rp_clip_coef).float().mean().item()
                            ]
                        if self.rp_norm_adv:
                            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                        pg_loss = torch.max(
                            -mb_adv * ratio,
                            -mb_adv * torch.clamp(ratio, 1 - self.rp_clip_coef, 1 + self.rp_clip_coef),
                        ).mean()
                        newvalue = newvalue.view(-1)
                        if self.rp_clip_vloss:
                            v_un = (newvalue - mb_ret) ** 2
                            v_cl = (
                                mb_val
                                + torch.clamp(newvalue - mb_val, -self.rp_clip_coef, self.rp_clip_coef)
                                - mb_ret
                            ) ** 2
                            v_loss = 0.5 * torch.max(v_un, v_cl).mean()
                        else:
                            v_loss = 0.5 * ((newvalue - mb_ret) ** 2).mean()
                        entropy_loss = entropy.mean() * self.rp_ent_coef
                        l1 = torch.mean(torch.abs(action_mean))
                        l2 = torch.mean(torch.square(action_mean))

                        # DPPO's critic warmup and ResiP's n_iterations_train_only_value both gate here.
                        in_warmup = self.itr < self.n_critic_warmup_itr
                        policy_loss = torch.zeros((), device=self.device)
                        if (self.itr + 1 > self.rp_train_only_value) and not in_warmup:
                            policy_loss = pg_loss - entropy_loss + self.rp_l1 * l1 + self.rp_l2 * l2
                        loss = policy_loss + v_loss * self.rp_vf_coef

                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            self.residual_policy.parameters(), self.rp_max_grad_norm
                        )
                        if not in_warmup:
                            self.actor_optimizer.step()
                        self.critic_optimizer.step()

                        if self.rp_target_kl is not None and approx_kl > self.rp_target_kl:
                            early_stop = True
                            break
                    if early_stop:
                        break

                y_pred, y_true = b_val.numpy(), b_ret.numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()

            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            run_results.append({"itr": self.itr, "step": cnt_train_step})
            if self.itr % self.log_freq == 0:
                elapsed = timer()
                run_results[-1]["time"] = elapsed
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr, commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | reward {avg_episode_reward:8.4f} | t:{elapsed:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "loss": loss.item(),
                                "pg loss": pg_loss.item(),
                                "value loss": v_loss.item(),
                                "entropy loss": entropy_loss.item(),
                                "residual l1": l1.item(),
                                "residual l2": l2.item(),
                                "approx kl": approx_kl.item(),
                                "ratio": ratio.mean().item(),
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                                "mean logstd": self.residual_policy.actor_logstd.mean().item(),
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                            },
                            step=self.itr, commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
