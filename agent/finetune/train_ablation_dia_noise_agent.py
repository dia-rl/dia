"""DIA with the inner advantage replaced by same-scale Gaussian noise."""

import logging

import numpy as np

from agent.finetune.train_ppo_dia_diffusion_agent import TrainPPODIADiffusionAgent

log = logging.getLogger(__name__)


class TrainAblationDiaNoiseAgent(TrainPPODIADiffusionAgent):

    def scale_match_inner(self, A_inner, advantages_outer):
        """Same std as the real scale-matched inner advantage, no structure."""
        scaled = super().scale_match_inner(A_inner, advantages_outer)
        tgt_std = float(scaled.std())
        noise = np.random.randn(*scaled.shape).astype(np.float32)
        noise *= tgt_std / (float(noise.std()) + 1e-8)
        return noise
