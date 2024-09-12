import numpy as np
from numpy.typing import NDArray
import torch
import pickle

from .simple_strategy import SimpleBiddingStrategy

from omegaconf import OmegaConf, MISSING
from torch import load

from bidding_train_env.utils import get_root_path, turn_off_grad
# from bidding_train_env.import_utils import get_actor
from ..agents import actor
class PlayerBiddingStrategy(SimpleBiddingStrategy):
    """
    Simple Strategy example for bidding.
    """
    def __init__(
            self,
            budget: float = 100.,
            name: str     = "PlayerStrategy",
            cpa: float    = 2,
            category: int = 1
        ):

        self.cpa = cpa
        self.budget = budget
        self.name = name
        self.category = category

        self.remaining_budget = budget

        config_path = get_root_path() / 'saved_models' / 'dt' / 'config.yaml'
        config = OmegaConf.load(config_path)

        agent =  getattr(actor,config.model.actor)(**config.model.actor_params)

        # actor = get_actor(config.model.actor, **config.model.actor_params)
        agent = agent.to(config.general.device)

        model_path = f"{get_root_path()}/{config.saved_models.actor}"

        print(model_path)

        state_dict = load(model_path, map_location=config.general.device)
        agent.load_state_dict(state_dict)

        turn_off_grad(agent)

        self.agent = agent
       
        

    def bidding(
            self,
            timeStepIndex          : int,
            pValues                : NDArray,
            pValueSigmas           : NDArray,
            historyPValueInfo      : list[NDArray],
            historyBid             : list[NDArray],
            historyAuctionResult   : list[NDArray],
            historyImpressionResult: list[NDArray],
            historyLeastWinningCost: list[NDArray],
        ) -> NDArray:

        bids, action, log_prob = self.get_bid_action(
            timeStepIndex,
            pValues,
            pValueSigmas,
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
        )
        
        alpha = action.cpu().detach().numpy()
        return alpha * pValues