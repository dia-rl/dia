"""
Evaluate a fine-tuned diffusion policy and record, per completed episode, the
END-EFFECTOR PATH LENGTH up to first success alongside the time-to-completion.

Transport's low-dim observation opens with [robot0_eef_pos(3), robot0_eef_quat(4),
robot0_gripper_qpos(2), robot1_eef_pos(3), ...], so the two end effectors sit at
dims 0:3 and 9:12. Observations arrive normalized to [-1, 1]; they are inverted
through the same normalization.npz the wrapper uses, so distances are in meters.
Positions are sampled once per policy step (act_steps env steps), which slightly
under-measures arc length, identically for every arm compared.
"""

import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent

EEF_SLICES = [(0, 3), (9, 12)]


class EvalDiffusionPathLenAgent(EvalAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ttc_tag = cfg.get("ttc_tag", "run")
        norm = np.load(cfg.normalization_path)
        self.obs_min = norm["obs_min"]
        self.obs_max = norm["obs_max"]

    def _unnorm(self, s):
        return (s + 1) / 2 * (self.obs_max - self.obs_min + 1e-6) + self.obs_min

    def run(self):
        timer = Timer()
        options_venv = [{} for _ in range(self.n_envs)]
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        obs_trajs = np.zeros((self.n_steps, self.n_envs, prev_obs_venv["state"].shape[-1]))

        for step in range(self.n_steps):
            if step % 20 == 0:
                print(f"[{self.ttc_tag}] step {step}/{self.n_steps}")
            obs_trajs[step] = prev_obs_venv["state"][:, -1]
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

        ttc_env_steps, path_len, ep_completed, ep_reward = [], [], [], []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start, end = env_steps[i], env_steps[i + 1]
                if end - start <= 1:
                    continue
                rt = reward_trajs[start:end, env_ind]
                ep_reward.append(float(rt.sum()))
                hit = np.where(rt > 0)[0]
                if len(hit) > 0:
                    ttc_env_steps.append(int(hit[0]) * self.act_steps)
                    seg = self._unnorm(obs_trajs[start:start + hit[0] + 1, env_ind])
                    d = 0.0
                    for a, b in EEF_SLICES:
                        d += float(np.linalg.norm(np.diff(seg[:, a:b], axis=0), axis=1).sum())
                    path_len.append(d)
                    ep_completed.append(True)
                else:
                    ep_completed.append(False)

        ttc_env_steps = np.array(ttc_env_steps)
        path_len = np.array(path_len)
        ep_completed = np.array(ep_completed)
        n_ep, n_done = len(ep_completed), int(np.sum(ep_completed))
        sr = n_done / max(n_ep, 1)
        out = os.path.join(self.logdir, f"pathlen_{self.ttc_tag}.npz")
        np.savez(out, ttc_env_steps=ttc_env_steps, path_len=path_len,
                 ep_completed=ep_completed, ep_reward=np.array(ep_reward),
                 act_steps=self.act_steps, max_episode_steps=self.max_episode_steps)
        log.info(
            f"[{self.ttc_tag}] episodes={n_ep} completed={n_done} SR={sr:.3f} | "
            f"ttc median={np.median(ttc_env_steps) if n_done else float('nan'):.1f} | "
            f"eef path length (m): median={np.median(path_len) if n_done else float('nan'):.3f} "
            f"mean={np.mean(path_len) if n_done else float('nan'):.3f} | saved {out} | t={timer():.0f}s"
        )
