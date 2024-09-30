from numpy.typing import NDArray

import numpy as np

from torch import Tensor
from sklearn.linear_model import LinearRegression

from .base_bidding_strategy import BasePolicyStrategy
from ..agents import Actor


class SigmaBiddingStrategy(BasePolicyStrategy):
    """
    Strategy with action that is bid = alpha * pValue + beta * sigma + theta
    """

    def __init__(
            self,
            actor: Actor,
            budget: float = 100.,
            name: str     = "SigmaStrategy",
            cpa: float    = 2.,
            category:int  = 1,
        ):
        super().__init__(actor, budget, name, cpa, category)

    def preprocess(
            self,
            timeStepIndex          : int,
            pValues                : NDArray,
            pValueSigmas           : NDArray,
            historyPValueInfo      : list[NDArray],
            historyBid             : list[NDArray],
            historyAuctionResult   : list[NDArray],
            historyImpressionResult: list[NDArray],
            historyLeastWinningCost: list[NDArray],
        ) -> Tensor:
        time_left = 1 - timeStepIndex / 48

        total_cost = sum(np.sum(array[1]) for array in historyBid)
        total_conversions = sum(np.sum(array) for array in historyImpressionResult)
        cpa_r = total_cost / total_conversions if total_conversions > 0 else 0.0
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
            historyBid_mean = [np.mean(array) for array in historyBid]
            historyLeastWinningCost_mean = [np.mean(array) for array in historyLeastWinningCost]
            historyPValueInfo_mean = [np.mean(array[:, 0]) for array in historyPValueInfo]
            historyImpressionResult_mean = [np.mean(array) for array in historyImpressionResult]
            historyAuctionResult_mean = [np.mean(array[:, 0]) for array in historyAuctionResult]

            historical_mean_bid = np.mean(historyBid_mean)
            historical_mean_least_winning_cost = np.mean(historyLeastWinningCost_mean)
            historical_mean_pValues = np.mean(historyPValueInfo_mean)
            historical_conversion_mean = np.mean(historyImpressionResult_mean)
            historical_xi_mean = np.mean(historyAuctionResult_mean)

            last_three_bid_mean = np.mean(historyBid_mean[-3:])
            last_three_LeastWinningCost_mean = np.mean(historyLeastWinningCost_mean[-3:])
            last_three_pValues_mean = np.mean(historyPValueInfo_mean[-3:])
            last_three_conversion_mean = np.mean(historyImpressionResult_mean[-3:])
            last_three_xi_mean = np.mean(historyAuctionResult_mean[-3:])

        current_pValues_mean = np.mean(pValues) if len(pValues) > 0 else 0.0
        current_pv_num = len(pValues)
        current_pValuesSigmas_mean = np.mean(pValueSigmas) if len(pValueSigmas) > 0 else 0.0

        last_three_pv_num_total = sum(
            len(pValues[0]) for pValues in historyPValueInfo[-3:]
        )
        historical_pv_num_total = sum(len(pValues[0]) for pValues in historyPValueInfo)

        return Tensor(
            [
                time_left,
                remaining_budget,
                current_pValues_mean,
                current_pValuesSigmas_mean,
                self.cpa,
                cpa_r,
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
        ).unsqueeze(0)


    def get_action(self, obs):
        action, log_prob, entropy = super().get_action(obs)

        return action.squeeze(-1).clamp(0), log_prob, entropy
    
    
    def action_to_bid(
            self,
            timeStepIndex          : int,
            pValues                : NDArray,
            pValueSigmas           : NDArray,
            historyPValueInfo      : list[NDArray],
            historyBid             : list[NDArray],
            historyAuctionResult   : list[NDArray],
            historyImpressionResult: list[NDArray],
            historyLeastWinningCost: list[NDArray],
            action: Tensor
        ) -> NDArray:
        alpha, beta, theta = action[:, 0], action[:, 1], action[:, 2]
        pValues = Tensor(pValues).to(action.device)
        pValueSigmas = Tensor(pValueSigmas).to(action.device)
        return alpha * pValues + beta * pValueSigmas + theta
    
    def bid_to_action(
        self,
        bids                   : NDArray,
        timeStepIndex          : int,
        pValues                : NDArray,
        pValueSigmas           : NDArray,
        historyPValueInfo      : list[NDArray],
        historyBid             : list[NDArray],
        historyAuctionResult   : list[NDArray],
        historyImpressionResult: list[NDArray],
        historyLeastWinningCost: list[NDArray],
    ) -> NDArray:
        # Linear regression model
        X = np.stack([pValues, pValueSigmas]).T        
        y = bids
        if np.isnan(y).mean() > 0:
            y = np.zeros_like(y)
        reg = LinearRegression().fit(X, y)
        alpha, beta = reg.coef_
        theta = reg.intercept_
        return np.array([alpha, beta, theta])