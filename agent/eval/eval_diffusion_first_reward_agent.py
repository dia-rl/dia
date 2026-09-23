"""
Evaluate a fine-tuned diffusion policy and record, per episode, the ENVIRONMENT STEP
at which the first positive reward is earned, and how many steps are spent in each
reward state.

The D3IL avoid (M5) per-step reward is

    -0.1                        if in collision, or outside x in [0.2, 0.8]
    1{mode_encoding[5]} + 1{y > 0.4}    otherwise

so a step pays 0, 1 or 2. Splitting the 1s needs to know which of the two terms is
on. `y > 0.4` is read straight off the end effector, observation dims 2:4
unnormalized, which the wrapper already passes out per environment step as
`full_obs`; the mode term is then whatever is left of the paid reward. Both terms are
recovered exactly, without re-implementing the env's reward.

Different from eval_diffusion_ttc_agent, which resolves only to the action chunk: the
reward returned by the multi-step wrapper is the sum over `act_steps` environment
steps, so on D3IL avoid (act_steps=4) it can only place the first reward within a
4-step window. This agent reads `info["step_rewards"]`, the per-environment-step
rewards the wrapper passes out when `pass_step_rewards` is set, so the answer is exact
in environment steps.

Sampling is stochastic, matching how every other avoid rollout statistic is produced.
D3IL avoid has no reset randomization, so with deterministic sampling every parallel
env would return the same trajectory.
"""

import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent


class EvalDiffusionFirstRewardAgent(EvalAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tag = cfg.get("tag", "run")
        self.deterministic = cfg.get("deterministic", False)
        self.goal_y = float(cfg.get("goal_y", 0.4))
        # observation dims 2:4 are the end effector; 0:2 are the previous commanded
        # target. The wrapper normalizes to [-1, 1] against the demonstration range.
        nz = np.load(cfg.normalization_path)
        self.obs_min, self.obs_max = nz["obs_min"], nz["obs_max"]

    def _unnorm_ee(self, o):
        lo, hi = self.obs_min[2:4], self.obs_max[2:4]
        return (o[..., 2:4] + 1) / 2 * (hi - lo) + lo

    def run(self):
        timer = Timer()
        options_venv = [{} for _ in range(self.n_envs)]
        self.model.eval()
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)

        # per env step, not per policy step
        T = self.n_steps * self.act_steps
        step_rewards = np.full((T, self.n_envs), np.nan, dtype=np.float32)
        ee_xy = np.full((T, self.n_envs, 2), np.nan, dtype=np.float32)
        done_venv = np.zeros(self.n_envs, dtype=bool)

        for step in range(self.n_steps):
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                samples = self.model(cond=cond, deterministic=self.deterministic)
                output_venv = samples.trajectories.cpu().numpy()
            action_venv = output_venv[:, : self.act_steps]
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            sr = self._extract_step_rewards(info_venv)
            fo = self._extract_full_obs(info_venv)
            lo = step * self.act_steps
            step_rewards[lo : lo + self.act_steps] = sr.T
            ee_xy[lo : lo + self.act_steps] = np.transpose(
                self._unnorm_ee(fo), (1, 0, 2)
            )
            # once an env is done its later entries are not part of this episode
            step_rewards[lo : lo + self.act_steps, done_venv] = np.nan
            ee_xy[lo : lo + self.act_steps, done_venv] = np.nan
            done_venv = done_venv | terminated_venv | truncated_venv
            prev_obs_venv = obs_venv

        first = np.full(self.n_envs, -1, dtype=np.int64)
        for e in range(self.n_envs):
            hit = np.where(step_rewards[:, e] > 0)[0]
            if len(hit):
                first[e] = int(hit[0])
        got = first >= 0
        ep_return = np.nansum(step_rewards, axis=0)

        # decompose the paid reward into its two terms
        live = ~np.isnan(step_rewards)
        penalty = live & (step_rewards < 0)
        paid = np.where(live & ~penalty, step_rewards, 0.0)
        goal = live & ~penalty & (ee_xy[..., 1] > self.goal_y)
        mode = (paid - goal.astype(np.float32)) > 0.5
        # the two terms must account for the paid reward exactly
        resid = np.abs(paid - goal.astype(np.float32) - mode.astype(np.float32))
        bad = int(np.nansum(resid > 1e-6))
        states = dict(
            steps_mode=(mode & ~goal).sum(0),
            steps_goal=(goal & ~mode).sum(0),
            steps_both=(mode & goal).sum(0),
            steps_penalty=penalty.sum(0),
            steps_zero=(live & ~penalty & ~mode & ~goal).sum(0),
            steps_mode_any=mode.sum(0),
            steps_goal_any=goal.sum(0),
        )

        out = os.path.join(self.logdir, f"firstrew_{self.tag}.npz")
        np.savez(
            out,
            first_reward_step=first,
            got_reward=got,
            episode_return=ep_return,
            step_rewards=step_rewards,
            ee_xy=ee_xy,
            decomposition_residual=bad,
            act_steps=self.act_steps,
            max_episode_steps=self.max_episode_steps,
            **states,
        )
        med = float(np.median(first[got])) if got.any() else float("nan")
        mean = float(np.mean(first[got])) if got.any() else float("nan")
        log.info(
            f"[{self.tag}] episodes={self.n_envs} with reward={int(got.sum())} | "
            f"first rewarded env step: median={med:.1f} mean={mean:.2f} | "
            f"avg return={float(np.mean(ep_return)):.2f} | saved {out} | t={timer():.0f}s"
        )
        print(
            f"RESULT {self.tag} n={self.n_envs} got={int(got.sum())} "
            f"median={med:.1f} mean={mean:.3f} return={float(np.mean(ep_return)):.2f} "
            f"mode_only={states['steps_mode'].mean():.2f} both={states['steps_both'].mean():.2f} "
            f"goal_only={states['steps_goal'].mean():.2f} pen={states['steps_penalty'].mean():.2f} "
            f"unexplained={bad}",
            flush=True,
        )

    def _extract_step_rewards(self, info_venv):
        """(n_envs, act_steps) of per-environment-step rewards.

        The vector env hands back either a list of per-env dicts or one batched dict,
        depending on the env family; accept both."""
        if isinstance(info_venv, (list, tuple)):
            return np.stack([np.asarray(i["step_rewards"]) for i in info_venv])
        return np.asarray(info_venv["step_rewards"])

    def _extract_full_obs(self, info_venv):
        """(n_envs, act_steps, obs_dim): the observation after each environment step
        of the chunk, which the wrapper keeps when pass_full_observations is set."""
        if isinstance(info_venv, (list, tuple)):
            return np.stack([np.asarray(i["full_obs"]["state"]) for i in info_venv])
        return np.asarray(info_venv["full_obs"]["state"])
