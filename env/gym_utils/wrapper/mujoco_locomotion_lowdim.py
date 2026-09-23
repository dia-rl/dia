"""
Environment wrapper for Gym environments (MuJoCo locomotion tasks) with state observations.

For consistency, we will use Dict{} for the observation space, with the key "state" for the state observation.
"""

import copy

import numpy as np
import gym
from gym import spaces


class MujocoLocomotionLowdimWrapper(gym.Env):
    def __init__(
        self,
        env,
        normalization_path,
        norm_range_floor=0.1,
        clip_obs=10.0,
    ):
        self.env = env
        # Guard against degenerate normalization: dims that are ~constant in the demos have
        # obs_max-obs_min ~ 0, so normalizing divides by ~0 and tiny runtime deviations explode
        # to |thousands|, blowing up single-head critics (see kitchen obs dim 13). Floor the
        # per-dim range and clip the normalized obs to a sane band.
        self.norm_range_floor = norm_range_floor
        self.clip_obs = clip_obs

        # setup spaces
        self.action_space = env.action_space
        normalization = np.load(normalization_path)
        self.obs_min = normalization["obs_min"]
        self.obs_max = normalization["obs_max"]
        self.action_min = normalization["action_min"]
        self.action_max = normalization["action_max"]

        self.observation_space = spaces.Dict()
        obs_example = self.env.reset()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space["state"] = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def get_sim_state(self):
        """Probe helper: everything a restored copy of this env needs to be the same env.

        The physics state alone is not enough for Franka Kitchen. It deletes a task
        from tasks_to_complete once solved and drives its reward off that set, and the
        robot keeps a cache of past observations it differences to get velocities, so
        both travel with the state.
        """
        u = self.env.unwrapped
        raw = u.sim.get_state()
        if hasattr(raw, "qpos"):
            # mujoco_py MjSimState is a ragged namedtuple (scalar time, two arrays,
            # an optional act, a dict), so np.asarray on it raises. Store the fields.
            physics = {
                "time": float(raw.time),
                "qpos": np.array(raw.qpos, copy=True),
                "qvel": np.array(raw.qvel, copy=True),
                "act": None if raw.act is None else np.array(raw.act, copy=True),
                "udd_state": copy.deepcopy(raw.udd_state) if raw.udd_state else {},
            }
        else:
            physics = np.asarray(raw).copy()
        state = {"physics": physics}
        if hasattr(u, "tasks_to_complete"):
            state["tasks_to_complete"] = set(u.tasks_to_complete)
        robot = getattr(u, "robot", None)
        if robot is not None and hasattr(robot, "_observation_cache"):
            state["obs_cache"] = copy.deepcopy(robot._observation_cache)
        return state

    def restore_sim_state(self, state):
        """Probe helper: inverse of get_sim_state. Returns the raw observation."""
        u = self.env.unwrapped
        physics = state["physics"]
        if isinstance(physics, dict):
            from mujoco_py import MjSimState

            u.sim.set_state(
                MjSimState(
                    time=physics["time"],
                    qpos=physics["qpos"],
                    qvel=physics["qvel"],
                    act=physics["act"],
                    udd_state=physics["udd_state"],
                )
            )
        else:
            u.sim.set_state(np.asarray(physics))
        u.sim.forward()
        if "tasks_to_complete" in state and hasattr(u, "tasks_to_complete"):
            u.tasks_to_complete = set(state["tasks_to_complete"])
        robot = getattr(u, "robot", None)
        if robot is not None and "obs_cache" in state:
            robot._observation_cache = copy.deepcopy(state["obs_cache"])
        return u._get_obs()

    def reset(self, **kwargs):
        """Ignore passed-in arguments like seed"""
        options = kwargs.get("options", {})
        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)
        sim_state = options.get("sim_state", None)
        if sim_state is not None:
            raw_obs = self.restore_sim_state(sim_state)
        else:
            raw_obs = self.env.reset()

        # normalize
        obs = self.normalize_obs(raw_obs)
        return {"state": obs}

    def normalize_obs(self, obs):
        denom = np.maximum(self.obs_max - self.obs_min, self.norm_range_floor)
        norm = 2 * ((obs - self.obs_min) / denom - 0.5)
        return np.clip(norm, -self.clip_obs, self.clip_obs)

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def step(self, action):
        raw_action = self.unnormalize_action(action)
        raw_obs, reward, done, info = self.env.step(raw_action)

        # normalize
        obs = self.normalize_obs(raw_obs)
        return {"state": obs}, reward, done, info

    def render(self, **kwargs):
        return self.env.render()
