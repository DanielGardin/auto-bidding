from numpy.typing import NDArray

from abc import ABC, abstractmethod

class BaseBiddingStrategy(ABC):
    """
    Base bidding strategy interface defining methods to be implemented.
    """

    def __init__(
            self,
            budget: float = 100.,
            name: str     = "BaseStrategy",
            cpa: float    = 2,
            category: int = 1
        ):
        """
        Initialize the bidding strategy.
        parameters:
            @budget: the advertiser's budget for a delivery period.
            @cpa: the CPA constraint of the advertiser.
            @category: the index of advertiser's industry category.

        """
        self.budget = budget
        self.remaining_budget = budget
        self.name = name
        self.cpa = cpa
        self.category = category


    def __repr__(self) -> str:
        return f"{self.name}(budget={self.budget}, cpa={self.cpa}, category={self.category})"


    def reset(self):
        """
        Reset the remaining budget to its initial state.
        """
        self.remaining_budget = self.budget


    @abstractmethod
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
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCost: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        pass

