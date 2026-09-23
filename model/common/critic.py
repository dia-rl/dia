"""
Critic networks.

"""

from typing import Union
import torch
from torch import nn
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.common.modules import SpatialEmb, RandomShiftsAug
from model.diffusion.modules import SinusoidalPosEmb


class CriticObs(torch.nn.Module):
    """State-only critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [cond_dim] + mlp_dims + [1]
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, cond: Union[dict, torch.Tensor]):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            or (B, num_feature) from ViT encoder
        """
        if isinstance(cond, dict):
            B = len(cond["state"])

            # flatten history
            state = cond["state"].view(B, -1)
        else:
            state = cond
        q1 = self.Q1(state)
        return q1


class CriticObsInnerState(torch.nn.Module):
    """Critic conditioned on (obs, x_t, denoising index t).

    DIA's V_inner: along the denoising chain it predicts the expected terminal Q,
    i.e. E[ Q(s, a_0) | x_t, t ], where a_0 is the action the chain emits.
    """

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        action_dim,
        horizon_steps,
        time_dim=32,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        **kwargs,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        input_dim = cond_dim + action_dim * horizon_steps + time_dim
        mlp_dims_full = [input_dim] + list(mlp_dims) + [1]
        model = ResidualMLP if residual_style else MLP
        self.Q1 = model(
            mlp_dims_full,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, cond: Union[dict, torch.Tensor], x_t: torch.Tensor, t):
        """
        cond: dict with key "state" (B, To, Do) or flat tensor (B, cond_dim)
        x_t: (B, Ta, Da) partially-denoised action chunk
        t: scalar int / (B,) tensor / int — denoising-step index
        """
        if isinstance(cond, dict):
            B = len(cond["state"])
            state = cond["state"].view(B, -1)
        else:
            state = cond
            B = state.shape[0]
        x_t_flat = x_t.view(B, -1)
        if isinstance(t, torch.Tensor):
            t_vec = t.to(device=state.device).float().reshape(-1)
            if t_vec.numel() == 1:
                t_vec = t_vec.expand(B)
        else:
            t_vec = torch.full((B,), float(t), device=state.device)
        t_emb = self.time_embedding(t_vec)
        inp = torch.cat([state, x_t_flat, t_emb], dim=-1)
        return self.Q1(inp)


class CriticObsAct(torch.nn.Module):
    """State-action critic network with N-head ensemble support.

    Backward compatible: double_q=True → 2 heads (Q1, Q2 attrs preserved).
    Override via n_heads=N for arbitrary ensemble. Forward returns a tuple of
    N (B,) tensors. Twin (N=2) callers receive (q1, q2) as before.
    """

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        action_dim,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        double_q=True,
        n_heads=None,
        dropout=0,
        **kwargs,
    ):
        super().__init__()
        if n_heads is None:
            n_heads = 2 if double_q else 1
        self.n_heads = int(n_heads)
        mlp_dims = [cond_dim + action_dim * action_steps] + mlp_dims + [1]
        model = ResidualMLP if residual_style else MLP
        self.heads = nn.ModuleList([
            model(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
                dropout=dropout,
            )
            for _ in range(self.n_heads)
        ])
        # Backward-compat aliases for code that references Q1/Q2 directly.
        if self.n_heads >= 1:
            self.Q1 = self.heads[0]
        if self.n_heads >= 2:
            self.Q2 = self.heads[1]

    def forward(self, cond: Union[dict, torch.Tensor], action):
        """Returns tuple of N (B,) tensors, one per ensemble head.

        cond may be a dict with key "state", or an already-flattened feature
        tensor (B, cond_dim) — the latter is what the ViT subclass passes after
        encoding the image.
        """
        if isinstance(cond, dict):
            B = len(cond["state"])
            state = cond["state"].view(B, -1)
        else:
            state = cond
            B = state.shape[0]
        action = action.view(B, -1)
        x = torch.cat((state, action), dim=-1)
        outs = tuple(h(x).squeeze(1) for h in self.heads)
        if self.n_heads == 1:
            return outs[0]
        return outs


class ViTEncodeMixin:
    """Shared ViT image encoding for the image critics.

    `encode` turns a {state, rgb} observation into the flat feature the MLP
    critics consume. Keeping it separate from `forward` lets a caller encode an
    observation once and reuse the result across several queries against the same
    observation, which is what DIA's inner critic needs: the image does not
    change with the denoising index, so encoding per index would multiply the ViT
    cost by K_ft for no benefit.
    """

    def _init_vit(
        self,
        backbone,
        cond_dim,
        img_cond_steps=1,
        spatial_emb=128,
        dropout=0,
        augment=False,
        num_img=1,
    ):
        self.backbone = backbone
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if num_img > 1:
            self.compress1 = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            self.compress2 = deepcopy(self.compress1)
        else:  # TODO: clean up
            self.compress = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment

    def encode(self, cond: dict, no_augment=False):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        returns: (B, spatial_emb * num_img + cond_dim)
        """
        B, T_rgb, C, H, W = cond["rgb"].shape

        # flatten history
        state = cond["state"].view(B, -1)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps :]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        # convert rgb to float32 for augmentation
        rgb = rgb.float()

        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment and not no_augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:  # single image
            if self.augment and not no_augment:
                rgb = self.aug(rgb)  # uint8 -> float32
            feat = self.backbone(rgb)
            feat = self.compress.forward(feat, state)
        return torch.cat([feat, state], dim=-1)


class ViTCritic(ViTEncodeMixin, CriticObs):
    """ViT + MLP, state only"""

    def __init__(
        self,
        backbone,
        cond_dim,
        img_cond_steps=1,
        spatial_emb=128,
        dropout=0,
        augment=False,
        num_img=1,
        **kwargs,
    ):
        # update input dim to mlp
        mlp_obs_dim = spatial_emb * num_img + cond_dim
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)
        self._init_vit(
            backbone,
            cond_dim,
            img_cond_steps=img_cond_steps,
            spatial_emb=spatial_emb,
            dropout=dropout,
            augment=augment,
            num_img=num_img,
        )

    def forward(self, cond: dict, no_augment=False):
        return CriticObs.forward(self, self.encode(cond, no_augment=no_augment))


class ViTCriticObsAct(ViTEncodeMixin, CriticObsAct):
    """ViT + MLP state-action critic: DIA's terminal Q over image observations.

    `cond` may be the raw {state, rgb} dict, or a feature already produced by
    `encode`, in which case the backbone is skipped.
    """

    def __init__(
        self,
        backbone,
        cond_dim,
        img_cond_steps=1,
        spatial_emb=128,
        dropout=0,
        augment=False,
        num_img=1,
        **kwargs,
    ):
        mlp_obs_dim = spatial_emb * num_img + cond_dim
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)
        self._init_vit(
            backbone,
            cond_dim,
            img_cond_steps=img_cond_steps,
            spatial_emb=spatial_emb,
            dropout=dropout,
            augment=augment,
            num_img=num_img,
        )

    def forward(self, cond, action, no_augment=False):
        feat = self.encode(cond, no_augment=no_augment) if isinstance(cond, dict) else cond
        return CriticObsAct.forward(self, feat, action)


class ViTCriticObsInnerState(ViTEncodeMixin, CriticObsInnerState):
    """ViT + MLP inner critic: DIA's V_k over image observations.

    `cond` may be the raw {state, rgb} dict, or a feature already produced by
    `encode`. The agent encodes once per environment step and passes the feature
    for every denoising index k.
    """

    def __init__(
        self,
        backbone,
        cond_dim,
        img_cond_steps=1,
        spatial_emb=128,
        dropout=0,
        augment=False,
        num_img=1,
        **kwargs,
    ):
        mlp_obs_dim = spatial_emb * num_img + cond_dim
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)
        self._init_vit(
            backbone,
            cond_dim,
            img_cond_steps=img_cond_steps,
            spatial_emb=spatial_emb,
            dropout=dropout,
            augment=augment,
            num_img=num_img,
        )

    def forward(self, cond, x_t, t, no_augment=False):
        feat = self.encode(cond, no_augment=no_augment) if isinstance(cond, dict) else cond
        return CriticObsInnerState.forward(self, feat, x_t, t)
