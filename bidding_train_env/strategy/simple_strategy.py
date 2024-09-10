from numpy.typing import NDArray

import numpy as np

from torch import Tensor

from .base_bidding_strategy import BasePolicyStrategy
from ..agents import Actor


class SimpleBiddingStrategy(BasePolicyStrategy):
    """
    Behavioral Cloning (bc) Strategy
    """

    def __init__(
            self,
            actor: Actor,
            budget=100,
            name="SimpleStrategy",
            cpa=2,
            category=1,
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
        remaining_budget = self.remaining_budget / (total_cost + self.remaining_budget)

        if timeStepIndex == 0:
            historical_mean_bid                = 0.
            last_three_bid_mean                = 0.

            historical_mean_least_winning_cost = 0.
            historical_mean_pValues            = 0.
            historical_conversion_mean         = 0.
            historical_xi_mean                 = 0.

            last_three_LeastWinningCost_mean   = 0.
            last_three_pValues_mean            = 0.
            last_three_conversion_mean         = 0.
            last_three_xi_mean                 = 0.

        else:
            historical_mean_bid = np.mean([np.mean(array) for array in historyBid])
            last_three_bid_mean = np.mean([np.mean(array) for array in historyBid[-3:]])

            historical_mean_least_winning_cost = np.mean([np.mean(array) for array in historyLeastWinningCost])
            historical_mean_pValues            = np.mean([np.mean(array[:, 0]) for array in historyPValueInfo])
            historical_conversion_mean         = np.mean([np.mean(array) for array in historyImpressionResult])
            historical_xi_mean                 = np.mean([np.mean(array[:, 0]) for array in historyAuctionResult])

            last_three_LeastWinningCost_mean = np.mean([np.mean(array) for array in historyLeastWinningCost[-3:]])
            last_three_pValues_mean          = np.mean([np.mean(array[:, 0]) for array in historyPValueInfo[-3:]])
            last_three_conversion_mean       = np.mean([np.mean(array) for array in historyImpressionResult[-3:]])
            last_three_xi_mean               = np.mean([np.mean(array[:, 0]) for array in historyAuctionResult[-3:]])

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        last_three_pv_num_total = sum(len(pValues[0]) for pValues in historyPValueInfo[-3:])
        historical_pv_num_total = sum(len(pValues[0]) for pValues in historyPValueInfo)

        return Tensor([
            time_left,
            remaining_budget,
            historical_mean_bid,
            last_three_bid_mean,
            historical_mean_least_winning_cost,
            historical_mean_pValues,
            historical_conversion_mean,
            historical_xi_mean,
            last_three_LeastWinningCost_mean,
            last_three_pValues_mean,
            last_three_conversion_mean,
            last_three_xi_mean,
            current_pValues_mean,
            current_pv_num / (500_000/48),
            last_three_pv_num_total / (500_000/3),
            historical_pv_num_total / 500_000
        ]).unsqueeze(0)


    def get_action(self, obs):
        action, log_prob, entropy = super().get_action(obs)

        return action.squeeze(-1), log_prob, entropy



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
        alpha = action.item()

        return alpha * pValues
