from typing import Optional, Protocol

from pathlib import Path

from .base_bidding_strategy import BaseBiddingStrategy as BaseBiddingStrategy
from .onlinelp_bidding_strategy import OnlineLpBiddingStrategy as OnlineLpBiddingStrategy
from .simple_strategy import SimpleBiddingStrategy as SimpleBiddingStrategy

from ..utils import get_root_path
from ..agents.actor import ContinuousDeterminisitcMLP

# Deploying strategy under PlayerBiddingStrategy

deploying_strategy = SimpleBiddingStrategy
actor              = ContinuousDeterminisitcMLP
parameters_path    = get_root_path() / "checkpoints/bc_2024-09-09-20:57:21/checkpoint_1000.pth"

actor_params       = {
    'input_dim' : 16,
    'output_dim': 1,
    'hidden_dim': [256, 256],
    'activation': 'relu'
}

class PlayerBiddingStrategy(SimpleBiddingStrategy):
    def __init__(
            self,
            budget=100,
            name="SimpleStrategy",
            cpa=2,
            category=1,
        ):

        actor = ContinuousDeterminisitcMLP(**actor_params)

        super().__init__(actor, budget, name, cpa, category)