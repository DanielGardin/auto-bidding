import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base_bidding_strategy import BaseBiddingStrategy


class CollectStrategy(BaseBiddingStrategy):
    """
    A strategy utilized to collect new state/action representations. It's only purpose is to wrap the methods from the original strategy and has a method to updated advertiser info.
    """

    def __init__(
        self,
        base_strategy : BaseBiddingStrategy,
        budget        : float = 100.0,
        name          : str = "CollectStrategy",
        cpa           : float = 2,
        category      : int = 1,
        data          : str = "new",
    ):

        self.base_strategy = base_strategy
        self.cpa = cpa
        self.budget = budget
        self.name = name
        self.category = category
        self.remaining_budget = budget
        if data == "new":
            self.advertiser_data = pd.read_csv(
                "data/traffic/new_efficient_repr/advertiser_data.csv", index_col=0
            )
        else:
            self.advertiser_data = pd.read_csv(
                "data/traffic/efficient_repr/advertiser_data.csv", index_col=0
            )

    def set_advertiser(self, advertiser: int):
        self.base_strategy.advertiser = advertiser
        self.base_strategy.cpa = self.advertiser_data.loc[advertiser, "CPAConstraint"]
        self.base_strategy.budget = self.advertiser_data.loc[advertiser, "budget"]
        self.base_strategy.category = self.advertiser_data.loc[
            advertiser, "advertiserCategoryIndex"
        ]
        self.base_strategy.remaining_budget = self.base_strategy.budget

        self.advertiser = advertiser
        self.cpa = self.base_strategy.cpa
        self.budget = self.base_strategy.budget
        self.category = self.base_strategy.category
        

    def bid_to_action(self, bids, **kwargs) -> NDArray:
        return self.base_strategy.bid_to_action(bids, **kwargs)
    
    def pay(self, cost: float):
        return self.base_strategy.pay(cost)

    def get_reward(self, **kwargs) -> float:
        return self.base_strategy.get_reward(**kwargs)

    def preprocess(self, **kwargs) -> NDArray:
        return self.base_strategy.preprocess(**kwargs)
    
    def state_names(self):
        return self.base_strategy.state_names()
    
    def action_names(self):
        return self.base_strategy.action_names()
    
    def reward_names(self):
        return self.base_strategy.reward_names()

    

