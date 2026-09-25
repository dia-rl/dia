"""DIA: Denoising Intermediate Advantage.

Adds a Q ensemble and an inner value over the denoising chain to PPODiffusion;
the actor loss is unchanged and the agent injects the combined advantage.

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension
"""
import copy
import torch
import logging

log = logging.getLogger(__name__)

from model.diffusion.diffusion_ppo import PPODiffusion


class PPODIADiffusion(PPODiffusion):
    def __init__(
        self,
        critic_q,                      # twin CriticObsAct (Q(s,a))
        critic_v_inner=None,           # optional CriticObsInnerState (V(s, x_k, k))
        q_aggregation="mean",          # "mean" (what every config in cfg/ uses) or "min"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.critic_q = critic_q.to(self.device)
        self.critic_q_target = copy.deepcopy(self.critic_q)
        for p in self.critic_q_target.parameters():
            p.requires_grad = False
        self.q_aggregation = str(q_aggregation)
        assert self.q_aggregation in ("min", "mean")
        if critic_v_inner is not None:
            self.critic_v_inner = critic_v_inner.to(self.device)
        else:
            self.critic_v_inner = None


    @torch.no_grad()
    def compute_q_safe(self, cond, action):
        """Aggregated Q via frozen Q_target across N heads. "mean" or "min". Returns (B,)."""
        qs = self.critic_q_target(cond, action)
        if not isinstance(qs, tuple):
            qs = (qs,)
        Q = torch.stack([q.view(-1) for q in qs], dim=0)   # (n_heads, B)
        if self.q_aggregation == "mean":
            return Q.mean(dim=0)
        return Q.min(dim=0).values

    # ---------- Q training: 1-step TD — N-head ensemble ----------
    def loss_critic_q(self, obs, next_obs, actions, rewards, terminated, gamma):
        with torch.no_grad():
            next_samples = self.forward(cond=next_obs, deterministic=False)
            next_actions = next_samples.trajectories
            nq = self.critic_q_target(next_obs, next_actions)
            if not isinstance(nq, tuple):
                nq = (nq,)
            nQ = torch.stack([q.view(-1) for q in nq], dim=0)   # (n_heads, B)
            if self.q_aggregation == "mean":
                next_q = nQ.mean(dim=0)
            else:
                next_q = nQ.min(dim=0).values
            mask = (1 - terminated.float()).view(-1)
            target = rewards.view(-1) + gamma * next_q * mask    # raw return units
        cq = self.critic_q(obs, actions)
        if not isinstance(cq, tuple):
            cq = (cq,)
        td_loss = sum(((q.view(-1) - target) ** 2).mean() for q in cq)
        return td_loss

    def update_critic_q_target(self, tau):
        for tp, sp in zip(self.critic_q_target.parameters(), self.critic_q.parameters()):
            tp.data.copy_(tp.data * (1 - tau) + sp.data * tau)

    # ---------- V_inner: value on inner MDP, target = the target Q at the emitted action x_K ----------
    def v_inner_forward(self, cond, x_chain_at_k, k_idx):
        """V_inner(s, x_k, k) -> (B,) scalar. Uses CriticObsInnerState architecture.
        k_idx may be int or (B,) long tensor."""
        out = self.critic_v_inner(cond, x_chain_at_k, k_idx)
        return out.view(-1)

    def loss_critic_v_inner(self, cond, x_chain_at_k, k_idx, target):
        """MSE regression: V_inner(s, x_k, k) -> target (detached the target Q at x_K)."""
        pred = self.v_inner_forward(cond, x_chain_at_k, k_idx)
        return ((pred - target.detach().view(-1)) ** 2).mean()
