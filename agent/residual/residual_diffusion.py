"""ResiP's composed policy: DPPO's frozen diffusion base plus a closed-loop residual."""

import logging

import torch
import torch.nn as nn

from model.diffusion.diffusion_vpg import VPGDiffusion

log = logging.getLogger(__name__)


class ResidualDiffusion(VPGDiffusion):
    """Frozen DPPO diffusion base plus ResiP's residual.

    Subclasses `VPGDiffusion`, the SIBLING of DPPO's `PPODiffusion`, rather than
    `PPODiffusion` itself. DIA extends `PPODiffusion` because it reuses DPPO's actor
    update and only adds critic terms; ResiP freezes the chain entirely, so
    `PPODiffusion.loss` would be dead code.

    One consequence to be aware of when reading a config: the four PPODiffusion-only
    keys carried over from the DPPO model block are accepted by `VPGDiffusion`'s
    `**kwargs` and have NO effect here, because the loss that reads them is never
    called:

        gamma_denoising, clip_ploss_coef, clip_ploss_coef_base, clip_ploss_coef_rate

    The `critic` block (DPPO's `CriticObs`) is likewise constructed but never trained
    or read; ResiP's value function is `residual_policy.critic`. All of these are kept
    so the config stays a literal copy of DPPO's.
    """

    def __init__(
        self,
        residual_policy,
        act_steps,
        obs_clip=3.0,
        action_clip=None,
        **kwargs,
    ):
        # ft_denoising_steps=0 freezes the whole chain on the base actor
        kwargs.setdefault("ft_denoising_steps", 0)
        super().__init__(**kwargs)
        assert self.ft_denoising_steps == 0, (
            "the base policy is frozen in ResiP; ft_denoising_steps must be 0 "
            f"(got {self.ft_denoising_steps})"
        )

        self.act_steps = act_steps
        self.obs_clip = obs_clip
        self.action_clip = action_clip

        for p in self.actor.parameters():
            p.requires_grad = False
        for p in self.actor_ft.parameters():
            p.requires_grad = False
        self.residual_policy = residual_policy.to(self.device)

        # per-environment chunk buffer and read pointer, allocated on first use
        self._chunk = None
        self._idx = None

        log.info(
            "ResidualDiffusion: frozen base %d params, residual %d params, "
            "chunk length %d, action_scale %s",
            sum(p.numel() for p in self.actor.parameters()),
            sum(p.numel() for p in self.residual_policy.parameters()),
            act_steps,
            self.residual_policy.action_scale,
        )

    def train(self, mode=True):
        """Keep the frozen base in eval mode whatever the agent does."""
        super().train(mode)
        self.actor.eval()
        self.actor_ft.eval()
        return self

    # ---------------------------------------------------------------- chunking
    def reset_chunks(self, n_envs):
        """Drop every cached chunk so the next call replans for all environments."""
        if self._chunk is None or len(self._chunk) != n_envs:
            self._chunk = torch.zeros(
                n_envs, self.act_steps, self.action_dim, device=self.device
            )
            self._idx = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self._idx.fill_(self.act_steps)

    @torch.no_grad()
    def base_action(self, cond, deterministic=False, force=None, advance=True):
        """The base action to execute now, replanning where a chunk is exhausted.

        Sampling goes through `VPGDiffusion.forward`, unmodified, so it is DPPO's.

        Args:
            cond: dict of observations, `state` of shape (B, To, Do)
            deterministic: passed through to DPPO's sampler; the agent sets it at
                evaluation exactly as the DPPO agent does
            force: bool tensor (B,), replan these environments regardless of the
                pointer, for environments that have just reset
            advance: move the pointer on; False leaves the queue untouched, for the
                bootstrap value at the end of an iteration
        """
        state = cond["state"]
        B = len(state)
        if self._chunk is None or len(self._chunk) != B:
            self.reset_chunks(B)

        need = self._idx >= self.act_steps
        if force is not None:
            need = need | torch.as_tensor(force, device=self.device).bool().reshape(-1)
        if need.any():
            sub = {
                k: torch.as_tensor(v)[need].float().to(self.device)
                for k, v in cond.items()
            }
            traj = super().forward(
                cond=sub, deterministic=deterministic, return_chain=False
            ).trajectories
            self._chunk[need] = traj[:, : self.act_steps].to(self._chunk.dtype)
            self._idx[need] = 0

        action = self._chunk[torch.arange(B, device=self.device), self._idx]
        if advance:
            self._idx += 1
        return action

    # ------------------------------------------------------------- composition
    def process_obs(self, state):
        """(B, To, Do) normalized observation to the (B, To*Do) the residual sees.

        The wrappers already normalize against the demonstration range; this is the
        flatten plus ResiP's out-of-range clamp (`src/behavior/residual_mlp.py`).
        """
        if not torch.is_tensor(state):
            state = torch.from_numpy(state)
        state = state.float().to(self.device).flatten(start_dim=1)
        if self.obs_clip is not None:
            state = torch.clamp(state, -self.obs_clip, self.obs_clip)
        return state

    def residual_obs(self, state, base_action):
        return torch.cat([self.process_obs(state), base_action], dim=-1)

    def compose(self, base_action, residual_action):
        """ResiP's composition, in the normalized action space the wrappers expect."""
        action = base_action + residual_action * self.residual_policy.action_scale
        if self.action_clip is not None:
            action = torch.clamp(action, -self.action_clip, self.action_clip)
        return action
