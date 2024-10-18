from typing import Callable, Sequence, Optional

import torch
import torch.nn as nn

from .base import Actor
from ..mlp import MLP

class DeterministicMLP(MLP, Actor):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            output_dim: int,
            activation: Callable[[torch.Tensor], torch.Tensor] | str = "relu",
            output_activation: Optional[Callable[[torch.Tensor], torch.Tensor] | str] = None,
            action_scale: float = 1.,
            action_interval: tuple[float, float] = (0., float('inf'))
        ):
        super().__init__(
            input_dim = input_dim,
            hidden_dims = hidden_dims,
            output_dim = output_dim,
            activation = activation,
            output_activation = output_activation
        )

        self.action_scale = action_scale
        self.action_interval = action_interval


    def get_action(
            self,
            obs: torch.Tensor,
            action: torch.Tensor | None = None,
            deterministic: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if action is None:
            action = self(obs)
            #action = torch.clamp(sampled_action * self.action_scale, *self.action_interval)

        return action, torch.tensor(0.), torch.tensor(0.)


class NormalStochasticMLP(MLP, Actor):
    stochastic = True

    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            output_dim: int,
            activation: Callable[[torch.Tensor], torch.Tensor] | str = "relu",
            log_std_lim: tuple[float, float] = (-20., 2.),
            mean_radius: Optional[float] = None,
            conditioned_std: bool = True
        ):
        self.output_dim = output_dim

        if conditioned_std:
            output_dim *= 2
        
        else:
            self.std_param = nn.Parameter(torch.zeros(output_dim, 1))

        super().__init__(
            input_dim = input_dim,
            hidden_dims = hidden_dims,
            output_dim = output_dim,
            activation = activation
        )

        self.log_std_lim     = log_std_lim
        self.mean_radius     = mean_radius
        self.conditioned_std = conditioned_std


    def get_action(
            self,
            obs: torch.Tensor,
            action: torch.Tensor | None = None,
            deterministic: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        n_samples = obs.size(0)

        output = self(obs)

        mean = output[:, :self.output_dim]

        if self.conditioned_std:
            log_std = output[:, self.output_dim:]

        else:
            log_std = self.std_param.expand(n_samples, self.output_dim)

        std = torch.clamp(log_std, *self.log_std_lim).exp()

        dist = torch.distributions.Normal(mean, std)

        if action is None:
            sampled_action = self.action_from_dist(dist, deterministic)

            if self.mean_radius is not None:
                sampled_action = self.mean_radius * torch.tanh(sampled_action)

        else:
            sampled_action = action

        log_prob = dist.log_prob(sampled_action).sum(dim=-1)    
        entropy  = dist.entropy()

        return sampled_action, log_prob, entropy