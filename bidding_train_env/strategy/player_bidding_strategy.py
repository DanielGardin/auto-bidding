import numpy as np
from numpy.typing import NDArray

from .base_bidding_strategy import BaseBiddingStrategy

class PlayerBiddingStrategy(BaseBiddingStrategy):
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
        super().__init__(budget, name, cpa, category)

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
