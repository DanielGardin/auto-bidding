from typing import Optional

import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class Critic(nn.Module, ABC):
    @abstractmethod
    def get_q_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get the Q-value of the given observation and action.

        ## Parameters
        - obs: torch.Tensor\\
        The observation to evaluate the Q-value on.

        - action: torch.Tensor\\
        The action to evaluate the Q-value on.

        ## Returns
        - q_value: torch.Tensor
        The Q-value of the given observation and action.
        """
        pass