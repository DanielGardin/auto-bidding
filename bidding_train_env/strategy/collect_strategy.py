import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sklearn.linear_model import LinearRegression

from .base_bidding_strategy import BaseBiddingStrategy


class CollectStrategy(BaseBiddingStrategy):
    """
    A strategy utilized to.
    """

    def __init__(
        self,
        base_strategy : BaseBiddingStrategy,
        budget: float = 100.0,
        name: str = "CollectStrategy",
        cpa: float = 2,
        category: int = 1,
    ):

        self.base_strategy = base_strategy
        self.cpa = cpa
        self.budget = budget
        self.name = name
        self.category = category
        self.remaining_budget = budget
        self.advertiser_data = pd.read_csv(
            "data/traffic/efficient_repr/advertiser_data.csv", index_col=0
        )

    def set_advertiser(self, advertiser: int):
        self.advertiser = advertiser
        self.cpa = self.advertiser_data.loc[self.advertiser, "CPAConstraint"]
        self.budget = self.advertiser_data.loc[self.advertiser, "budget"]
        self.category = self.advertiser_data.loc[
            self.advertiser, "advertiserCategoryIndex"
        ]
        self.remaining_budget = self.budget

    def bid_to_action(
        self,
        bids: NDArray,
        timeStepIndex: int,
        pValues: NDArray,
        pValueSigmas: NDArray,
        historyPValueInfo: list[NDArray],
        historyBid: list[NDArray],
        historyAuctionResult: list[NDArray],
        historyImpressionResult: list[NDArray],
        historyLeastWinningCost: list[NDArray],
    ) -> NDArray:

        #alpha = bids.sum() / pValues.sum()
        #return np.array([alpha])

        # Linear regression model
        X = np.stack([pValues, pValueSigmas]).T
        y = bids
        reg = LinearRegression().fit(X, y)
        alpha, beta = reg.coef_
        theta = reg.intercept_
        return np.array([alpha, beta, theta])

    def get_reward(
        self,
        timeStepIndex: int,
        pValues: NDArray,
        pValueSigmas: NDArray,
        historyPValueInfo: list[NDArray],
        historyBid: list[NDArray],
        historyAuctionResult: list[NDArray],
        historyImpressionResult: list[NDArray],
        historyLeastWinningCost: list[NDArray],
    ) -> float:
        """Update if design a different reward"""
        return super().get_reward(
            timeStepIndex,
            pValues,
            pValueSigmas,
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
        )

    def preprocess(self, **kwargs) -> NDArray:
        return self.base_strategy.preprocess(**kwargs)
