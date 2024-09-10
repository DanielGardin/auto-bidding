from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            output_dim: int,
            activation: Callable[[torch.Tensor], torch.Tensor] | str = "relu"
        ):
        super(MLP, self).__init__()

        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))

            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)

        if isinstance(activation, str):
            if not hasattr(F, activation):
                raise ValueError(f"Invalid activation function: {activation}")
            
            self.activation = getattr(F, activation)
        
        else:
            self.activation = activation


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        x = self.output_layer(x)

        return x