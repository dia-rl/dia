"""ResiP's residual actor and critic over [observation, base action]."""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

log = logging.getLogger(__name__)


def layer_init(layer, nonlinearity="ReLU", std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        if nonlinearity == "ReLU":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "SiLU":
            # upstream deliberately reuses the relu gain for Swish
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "Tanh":
            nn.init.orthogonal_(layer.weight, std)
        else:
            nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def build_mlp(
    input_dim,
    mlp_dims,
    output_dim,
    activation_type,
    output_std=1.0,
    bias_on_last_layer=True,
    last_layer_bias_const=0.0,
):
    act = getattr(nn, activation_type)
    layers = [
        layer_init(nn.Linear(input_dim, mlp_dims[0]), nonlinearity=activation_type),
        act(),
    ]
    for i in range(1, len(mlp_dims)):
        layers.append(
            layer_init(
                nn.Linear(mlp_dims[i - 1], mlp_dims[i]), nonlinearity=activation_type
            )
        )
        layers.append(act())
    layers.append(
        layer_init(
            nn.Linear(mlp_dims[-1], output_dim, bias=bias_on_last_layer),
            std=output_std,
            nonlinearity="Tanh",
            bias_const=last_layer_bias_const,
        )
    )
    return nn.Sequential(*layers)


class ResidualPolicy(nn.Module):
    """Gaussian actor and value critic over [observation, base action].

    Args:
        cond_dim: width of the observation the residual conditions on, flattened
            over the conditioning steps
        action_dim: action width; the residual has the same width
    """

    def __init__(
        self,
        cond_dim,
        action_dim,
        actor_mlp_dims=[256, 256],
        critic_mlp_dims=[256, 256],
        actor_activation_type="ReLU",
        critic_activation_type="ReLU",
        init_logstd=-1.0,
        learn_std=False,
        action_head_std=0.0,
        action_scale=0.1,
        critic_last_layer_std=0.25,
        critic_last_layer_bias_const=0.25,
        critic_last_layer_activation=None,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        # the residual sees the observation with the base action appended
        self.obs_dim = int(cond_dim) + self.action_dim
        self.action_scale = action_scale

        self.actor_mean = build_mlp(
            input_dim=self.obs_dim,
            mlp_dims=list(actor_mlp_dims),
            output_dim=self.action_dim,
            activation_type=actor_activation_type,
            output_std=action_head_std,
            bias_on_last_layer=False,
        )
        self.critic = build_mlp(
            input_dim=self.obs_dim,
            mlp_dims=list(critic_mlp_dims),
            output_dim=1,
            activation_type=critic_activation_type,
            output_std=critic_last_layer_std,
            bias_on_last_layer=True,
            last_layer_bias_const=critic_last_layer_bias_const,
        )
        if critic_last_layer_activation is not None:
            self.critic.add_module(
                "output_activation", getattr(nn, critic_last_layer_activation)()
            )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, self.action_dim) * init_logstd, requires_grad=learn_std
        )

        log.info(
            "ResidualPolicy: obs_dim=%d action_dim=%d action_scale=%s "
            "actor %s (%d params) critic %s (%d params) learn_std=%s init_logstd=%s",
            self.obs_dim,
            self.action_dim,
            action_scale,
            list(actor_mlp_dims),
            sum(p.numel() for p in self.actor_parameters),
            list(critic_mlp_dims),
            sum(p.numel() for p in self.critic_parameters),
            learn_std,
            init_logstd,
        )

    @property
    def actor_parameters(self):
        return [p for n, p in self.named_parameters() if "critic" not in n]

    @property
    def critic_parameters(self):
        return [p for n, p in self.named_parameters() if "critic" in n]

    def get_value(self, nobs):
        return self.critic(nobs)

    def get_action_and_value(self, nobs, action=None):
        action_mean = self.actor_mean(nobs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(dim=1),
            probs.entropy().sum(dim=1),
            self.critic(nobs),
            action_mean,
        )
