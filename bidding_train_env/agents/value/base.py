from typing import Optional

import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class Value(nn.Module, ABC):
    @abstractmethod
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the value of the given observation.

        ## Parameters
        - obs: torch.Tensor\\
        The observation to evaluate the value on.

        ## Returns
        - value: torch.Tensor
        The value of the given observation.
        """
        pass