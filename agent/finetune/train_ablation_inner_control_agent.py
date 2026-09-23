"""Control arm for DIA: keep the inner advantage's shape, destroy what it knows."""

import logging

import numpy as np

from agent.finetune.train_ppo_dia_diffusion_agent import TrainPPODIADiffusionAgent

log = logging.getLogger(__name__)

MODES = ("delta_gaussian", "shuffle_cross_sample")


class TrainAblationInnerControlAgent(TrainPPODIADiffusionAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.inner_control_mode = str(cfg.train.get("inner_control_mode", "delta_gaussian"))
        if self.inner_control_mode not in MODES:
            raise ValueError(
                f"inner_control_mode={self.inner_control_mode!r} is not one of {MODES}"
            )
        # Own RNG stream, so the substitution does not depend on other draws from numpy's global one.
        self.inner_control_rng = np.random.default_rng(int(cfg.seed) + 20260831)
        log.info(
            f"DIA inner-advantage control: mode={self.inner_control_mode}, "
            f"alpha={self.v_inner_alpha}, lambda_inner={self.v_inner_lambda} "
            f"(V_inner is still trained; only the advantage built from it is replaced)"
        )

    def _deltas(self, A):
        """Invert the inner GAE: d_k = A[k] - lam * A[k+1], with A[K] = 0."""
        lam, K = self.v_inner_lambda, A.shape[2]
        d = np.empty_like(A)
        for k in range(K):
            nxt = A[:, :, k + 1] if k + 1 < K else 0.0
            d[:, :, k] = A[:, :, k] - lam * nxt
        return d

    def _gae(self, d):
        """Re-run the inner GAE over substituted increments."""
        lam, K = self.v_inner_lambda, d.shape[2]
        A = np.empty_like(d)
        last = np.zeros(d.shape[:2], dtype=d.dtype)
        for k in reversed(range(K)):
            A[:, :, k] = last = d[:, :, k] + lam * last
        return A

    def scale_match_inner(self, A_inner, advantages_outer):
        """Substitute, then hand the result to DIA's own scale match."""
        A_inner = np.asarray(A_inner)
        S, E, K = A_inner.shape
        rng, mode = self.inner_control_rng, self.inner_control_mode

        if mode == "delta_gaussian":
            # Randomize the increments, not the advantage, so the per-k envelope is rebuilt by the recursion.
            d = self._deltas(A_inner)
            sub = rng.standard_normal(d.shape).astype(d.dtype) * float(d.std())
            out = self._gae(sub)
            # then through DIA's scale match exactly as the real arm does
            real = np.asarray(super().scale_match_inner(A_inner, advantages_outer))
            out = np.asarray(super().scale_match_inner(out, advantages_outer))
            return out.astype(real.dtype)

        # shuffle_cross_sample: scale-match first, then permute across transitions at each fixed k.
        real = np.asarray(super().scale_match_inner(A_inner, advantages_outer))
        out = np.empty_like(real)
        for k in range(K):
            flat = real[:, :, k].reshape(-1).copy()
            rng.shuffle(flat)
            out[:, :, k] = flat.reshape(S, E)
        return out.astype(real.dtype)

