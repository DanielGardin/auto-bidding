from numpy.typing import NDArray

import pandas as pd
import os

from .base_bidding_strategy import BaseBiddingStrategy


class OnlineLpBiddingStrategy(BaseBiddingStrategy):
    """
    OnlineLpBidding Strategy
    """

    def __init__(
            self,
            budget=100,
            name="OnlineLpStrategy",
            cpa=2,
            category=1
        ):
        super().__init__(budget, name, cpa, category)
        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        model_path = os.path.join(dir_name, "saved_model", "onlineLpTest", f"period.csv")
        self.category = category

        self.model = pd.read_csv(model_path)

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

        tem = self.model[
            (self.model["timeStepIndex"] == timeStepIndex) & (self.model["advertiserCategoryIndex"] == self.category)]
        alpha = self.cpa
        if (len(tem) == 0):
            pass
        else:
            def find_first_cpa_above_budget(df, budget):
                filtered_df = df[df['cum_cost'] > budget]

                if not filtered_df.empty:
                    return filtered_df.iloc[0]['realCPA']
                else:
                    return None

            res = find_first_cpa_above_budget(tem, self.remaining_budget)
            if res is None:
                pass
            else:
                alpha = res

        alpha = min(self.cpa * 1.5, alpha)
        bids = alpha * pValues
        return bids
