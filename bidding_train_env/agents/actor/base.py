from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Distribution

from abc import ABC, abstractmethod

class Actor(nn.Module, ABC):
    stochastic = False

    def __init__(self):
        super().__init__()


    def is_stochastic(self):
        return self.stochastic

    def reset(self):
        pass

    def callback(self, reward: float):
        pass


    def action_from_dist(self, dist: Distribution, deterministic: bool) -> torch.Tensor:
        if self.stochastic and not deterministic:
            return dist.rsample()

        return dist.mode


    @abstractmethod
    def get_action(
            self,
            obs: torch.Tensor,
            action: Optional[torch.Tensor] = None,
            deterministic: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get an action from the agent, with the option to pass in an action to evaluate.
        Returns the action, its log probability, and the entropy of the agent.

        ## Parameters
        - obs: torch.Tensor\\
        The observation to evaluate the action on.

        - action: Optional[Any]\\
        The action to evaluate. If None, the agent will sample one.
        
        - deterministic: bool\\
        Whether to sample a deterministic action or not. Only used for stochastic agents.

        ## Returns
        - action: torch.Tensor\\
        The given or sampled action by the agent.

        - log_prob: torch.Tensor\\
        The action's log probability.

        - entropy: torch.Tensor\\
        The entropy of the agent.
        """
        pass

