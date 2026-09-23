"""
Evaluate a fine-tuned diffusion policy and record per-episode TIME-TO-COMPLETION
(first env-step at which the sparse success reward fires). Used to test whether DIA
reaches the goal earlier than DPPO on transport at matched success rate.
"""

import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent


class EvalDiffusionTTCAgent(EvalAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ttc_tag = cfg.get("ttc_tag", "run")

    def run(self):
        timer = Timer()
        options_venv = [{} for _ in range(self.n_envs)]
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))

        for step in range(self.n_steps):
            if step % 20 == 0:
                print(f"[{self.ttc_tag}] step {step}/{self.n_steps}")
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                }
                samples = self.model(cond=cond, deterministic=True)
                output_venv = samples.trajectories.cpu().numpy()
            action_venv = output_venv[:, : self.act_steps]
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv
            prev_obs_venv = obs_venv

        # split into episodes; per episode record first success step (in POLICY steps) and env-steps
        ttc_env_steps = []     # time to first success, completed episodes only
        ep_completed = []      # bool per episode
        ep_reward = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start, end = env_steps[i], env_steps[i + 1]
                if end - start <= 1:
                    continue
                rt = reward_trajs[start:end, env_ind]          # per-policy-step reward
                ep_reward.append(float(rt.sum()))
                hit = np.where(rt > 0)[0]
                if len(hit) > 0:
                    ttc_env_steps.append(int(hit[0]) * self.act_steps)  # first success in env-steps
                    ep_completed.append(True)
                else:
                    ep_completed.append(False)

        ttc_env_steps = np.array(ttc_env_steps)
        ep_completed = np.array(ep_completed)
        n_ep = len(ep_completed)
        n_done = int(ep_completed.sum())
        sr = n_done / max(n_ep, 1)
        out = os.path.join(self.logdir, f"ttc_{self.ttc_tag}.npz")
        np.savez(out, ttc_env_steps=ttc_env_steps, ep_completed=ep_completed,
                 ep_reward=np.array(ep_reward), act_steps=self.act_steps,
                 max_episode_steps=self.max_episode_steps)
        med = float(np.median(ttc_env_steps)) if n_done else float("nan")
        mean = float(np.mean(ttc_env_steps)) if n_done else float("nan")
        log.info(
            f"[{self.ttc_tag}] episodes={n_ep} completed={n_done} SR={sr:.3f} | "
            f"time-to-completion (env steps): median={med:.1f} mean={mean:.1f} | "
            f"avg ep reward={np.mean(ep_reward):.2f} | saved {out} | t={timer():.0f}s"
        )
