from typing import Optional, Protocol

from pathlib import Path

import torch

from .base_bidding_strategy import BaseBiddingStrategy as BaseBiddingStrategy
from .onlinelp_bidding_strategy import OnlineLpBiddingStrategy as OnlineLpBiddingStrategy
from .simple_strategy import SimpleBiddingStrategy as SimpleBiddingStrategy
from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy

from ..utils import get_root_path
from ..agents.actor import ContinuousDeterminisitcMLP, ContinousStochasticMLP

# Deploying strategy under PlayerBiddingStrategy

deploying_strategy = SimpleBiddingStrategy
actor              = ContinuousDeterminisitcMLP
parameters_path    = get_root_path() / "checkpoints/iql_2024-09-11-12:08:25/actor_checkpoint_12.pth"


actor_params       = {
    'input_dim' : 16,
    'output_dim': 1,
    'hidden_dims': [256, 256],
    'activation': 'relu'
}

# class PlayerBiddingStrategy(SimpleBiddingStrategy):
#     def __init__(
#             self,
#             budget=100,
#             name="SimpleStrategy",
#             cpa=2,
#             category=1,
#         ):

#         actor = ContinousStochasticMLP(**actor_params)
#         actor.load_state_dict(torch.load(parameters_path, map_location='cpu'))

#         super().__init__(actor, budget, name, cpa, category)