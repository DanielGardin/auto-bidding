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
        budget: float = 100.0,
        name: str = "CollectStrategy",
        cpa: float = 2,
        category: int = 1,
    ):

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

    def bidding(
        self,
        timeStepIndex: int,
        pValues: NDArray,
        pValueSigmas: NDArray,
        historyPValueInfo: list[NDArray],
        historyBid: list[NDArray],
        historyAuctionResult: list[NDArray],
        historyImpressionResult: list[NDArray],
        historyLeastWinningCost: list[NDArray],
    ) -> NDArray:
        """Not used in this strategy"""
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

    def preprocess(
        self,
        timeStepIndex: int,
        pValues: NDArray,
        pValueSigmas: NDArray,
        historyPValueInfo: list[NDArray],
        historyBid: list[NDArray],
        historyAuctionResult: list[NDArray],
        historyImpressionResult: list[NDArray],
        historyLeastWinningCost: list[NDArray],
    ) -> NDArray:
        """Update if design a different state representation. State may use self attributes."""
        time_left = 1 - timeStepIndex / 48

        total_cost = sum(np.sum(array[1]) for array in historyBid)
        remaining_budget = self.remaining_budget / (total_cost + self.remaining_budget)

        if timeStepIndex == 0:
            historical_mean_bid = 0.0
            last_three_bid_mean = 0.0

            historical_mean_least_winning_cost = 0.0
            historical_mean_pValues = 0.0
            historical_conversion_mean = 0.0
            historical_xi_mean = 0.0

            last_three_LeastWinningCost_mean = 0.0
            last_three_pValues_mean = 0.0
            last_three_conversion_mean = 0.0
            last_three_xi_mean = 0.0

        else:
            historical_mean_bid = np.mean([np.mean(array) for array in historyBid])
            last_three_bid_mean = np.mean([np.mean(array) for array in historyBid[-3:]])

            historical_mean_least_winning_cost = np.mean(
                [np.mean(array) for array in historyLeastWinningCost]
            )
            historical_mean_pValues = np.mean(
                [np.mean(array[:, 0]) for array in historyPValueInfo]
            )
            historical_conversion_mean = np.mean(
                [np.mean(array) for array in historyImpressionResult]
            )
            historical_xi_mean = np.mean(
                [np.mean(array[:, 0]) for array in historyAuctionResult]
            )

            last_three_LeastWinningCost_mean = np.mean(
                [np.mean(array) for array in historyLeastWinningCost[-3:]]
            )
            last_three_pValues_mean = np.mean(
                [np.mean(array[:, 0]) for array in historyPValueInfo[-3:]]
            )
            last_three_conversion_mean = np.mean(
                [np.mean(array) for array in historyImpressionResult[-3:]]
            )
            last_three_xi_mean = np.mean(
                [np.mean(array[:, 0]) for array in historyAuctionResult[-3:]]
            )

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)
        current_pValuesSigmas_mean = np.mean(pValueSigmas)

        last_three_pv_num_total = sum(
            len(pValues[0]) for pValues in historyPValueInfo[-3:]
        )
        historical_pv_num_total = sum(len(pValues[0]) for pValues in historyPValueInfo)

        return np.array(
            [
                time_left,
                remaining_budget,
                current_pValues_mean,
                current_pValuesSigmas_mean,
                self.cpa,
                self.budget,
            ]
            + [1 if self.category == i else 0 for i in range(6)]
            + [
                historical_mean_bid,
                historical_mean_least_winning_cost,
                historical_mean_pValues,
                historical_conversion_mean,
                historical_xi_mean,
                last_three_bid_mean,
                last_three_LeastWinningCost_mean,
                last_three_pValues_mean,
                last_three_conversion_mean,
                last_three_xi_mean,
                current_pv_num / (500_000 / 48),
                last_three_pv_num_total / (500_000 / 3),
                historical_pv_num_total / 500_000,
            ]
        )

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
