from typing import Callable, Sequence

import torch
import torch.nn as nn

from .base import Value
from ..mlp import MLP

class MLP(MLP, Value):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            activation: Callable[[torch.Tensor], torch.Tensor] | str = "relu"
        ):
        super().__init__(
            input_dim = input_dim,
            hidden_dims = hidden_dims,
            output_dim = 1,
            activation = activation
        )

    def get_value(
            self,
            obs: torch.Tensor
        ) -> torch.Tensor:

        return self.forward(obs).squeeze()