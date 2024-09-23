from typing import Sequence, Callable

import torch
import torch.nn as nn

from math import prod

from .base import Critic
from ..mlp import MLP

class QEmbedMLP(Critic):
    def __init__(
            self,
            observation_shape: Sequence[int],
            action_shape: Sequence[int],
            embedding_dim: int,
            hidden_dims: Sequence[int],
            activation: Callable[[torch.Tensor], torch.Tensor] | str = "relu"
        ):
        super().__init__()

        self.obs_dim = prod(observation_shape)
        self.act_dim = prod(action_shape)

        self.obs_embed = nn.Linear(self.obs_dim, embedding_dim)
        self.act_embed = nn.Linear(self.act_dim, embedding_dim)

        self.mlp = MLP(
            input_dim = 2 * embedding_dim,
            hidden_dims = hidden_dims,
            output_dim = 1,
            activation = activation
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        obs = self.obs_embed(obs.view(-1, self.obs_dim))
        act = self.act_embed(act.view(-1, self.act_dim))

        x = torch.cat([obs, act], dim=-1)

        return self.mlp(x)

    def get_q_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self(obs, action).squeeze()