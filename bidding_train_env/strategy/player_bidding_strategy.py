import numpy as np
from numpy.typing import NDArray

from .base_bidding_strategy import BaseBiddingStrategy

class PlayerBiddingStrategy(BaseBiddingStrategy):
    """
    Simple Strategy example for bidding.
    """
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

        return self.cpa * pValues
