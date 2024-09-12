from typing import Optional
from numpy.typing import NDArray

from abc import ABC, abstractmethod

from ..strategy import BaseBiddingStrategy

class BiddingEnv(ABC):
    def __init__(
            self,
            strategy: BaseBiddingStrategy,
            period: int
        ):
        # Agent internal state
        self.remaining_budget = strategy.budget

        self.history = {
            'historyPValueInfo'       : [],
            'historyBid'              : [],
            'historyAuctionResult'    : [],
            'historyImpressionResult' : [],
            'historyLeastWinningCost' : [],
        }

        self.set_period(period)
    

    def set_period(self, period: int):
        self.current_period = period

    def set_strategy(self, strategy: BaseBiddingStrategy):
        self.strategy = strategy

    def __repr__(self):
        return f"BiddingEnv"

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the environment is in a terminal state.
        """
        pass


    @abstractmethod
    def reset(self) -> tuple[dict, dict]:
        """
        Reset the environment.
        """
        pass


    @abstractmethod
    def step(self, bids: NDArray) -> tuple[dict, float, bool, dict]:
        """
        Take a step in the environment.
        """
        pass