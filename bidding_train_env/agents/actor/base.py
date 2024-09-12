from typing import Optional

import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class Actor(nn.Module, ABC):
    def reset(self):
        pass

    def callback(self, reward: float):
        pass

    @abstractmethod
    def get_action(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get an action from the agent, with the option to pass in an action to evaluate.
        Returns the action, its log probability, and the entropy of the agent.

        ## Parameters
        - obs: torch.Tensor\\
        The observation to evaluate the action on.

        - action: Optional[Any]\\
        The action to evaluate. If None, the agent will sample one.
        
        ## Returns
        - action: torch.Tensor
        The given or sampled action by the agent.

        - log_prob: torch.Tensor
        The action's log probability.

        - entropy: torch.Tensor
        The entropy of the agent.
        """
        pass

