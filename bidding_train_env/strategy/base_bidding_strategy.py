from numpy.typing import NDArray

from abc import ABC, abstractmethod
from torch import Tensor
import numpy as np

from ..agents import Actor

class BaseBiddingStrategy:
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


    def pay(self, cost: float):
        """
        Pay the cost from the remaining budget.
        """
        self.remaining_budget -= cost


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
        ...
    

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
        ...


    def get_bid_action(
            self,
            timeStepIndex          : int,
            pValues                : NDArray,
            pValueSigmas           : NDArray,
            historyPValueInfo      : list[NDArray],
            historyBid             : list[NDArray],
            historyAuctionResult   : list[NDArray],
            historyImpressionResult: list[NDArray],
            historyLeastWinningCost: list[NDArray],
        ) -> tuple[NDArray, Tensor, Tensor]:
        # Default bidding strategy will be removed in the future in favor to abc
        return self.cpa * pValues, Tensor(0.), Tensor(0.)


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
        bids, _, _ = self.get_bid_action(
            timeStepIndex,
            pValues,
            pValueSigmas,
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
        )

        return bids

    
    def get_reward(
            self,
            timeStepIndex          : int,
            pValues                : NDArray,
            pValueSigmas           : NDArray,
            historyPValueInfo      : list[NDArray],
            historyBid             : list[NDArray],
            historyAuctionResult   : list[NDArray],
            historyImpressionResult: list[NDArray],
            historyLeastWinningCost: list[NDArray],
    ) -> float:
        
        if len(historyImpressionResult) == 0:
            return 0.

        else:
            return np.sum(historyImpressionResult[-1]).astype(float)


class BasePolicyStrategy(BaseBiddingStrategy, ABC):
    """
    Base policy strategy interface defining methods to be implemented.
    """

    def __init__(
            self,
            agent: Actor,
            budget: float = 100.,
            name: str     = "BasePolicyStrategy",
            cpa: float    = 2.,
            category: int = 1
        ):
        """
        Initialize the policy strategy.
        parameters:
            @budget: the advertiser's budget for a delivery period.
            @cpa: the CPA constraint of the advertiser.
            @category: the index of advertiser's industry category.

        """
        super().__init__(budget, name, cpa, category)

        self.agent = agent


    @property
    def device(self):
        return next(self.agent.parameters()).device


    def reset(self):
        """
        Reset the remaining budget to its initial state.
        """
        super().reset()
        self.agent.reset()


    @abstractmethod
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
        """
        Preprocess the observation data into a tensor for the agent.
        """

        pass


    def get_action(self, obs) -> tuple[Tensor, Tensor, Tensor]:
        return self.agent.get_action(obs)


    @abstractmethod
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
        """
        Given the observation and an action sampled from the agent, convert the action to bids.
        """

        pass


    def get_bid_action(
            self,
            timeStepIndex          : int,
            pValues                : NDArray,
            pValueSigmas           : NDArray,
            historyPValueInfo      : list[NDArray],
            historyBid             : list[NDArray],
            historyAuctionResult   : list[NDArray],
            historyImpressionResult: list[NDArray],
            historyLeastWinningCost: list[NDArray],
        ) -> tuple[NDArray, Tensor, Tensor]:

        previous_reward = self.get_reward(
            timeStepIndex,
            pValues,
            pValueSigmas,
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
        )

        obs = self.preprocess(
            timeStepIndex,
            pValues,
            pValueSigmas,
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
        ).to(self.device)

        self.agent.callback(previous_reward)

        action, log_prob, entropy = self.get_action(obs)

        bids = self.action_to_bid(
            timeStepIndex,
            pValues,
            pValueSigmas,
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
            action
        )

        return bids, action, log_prob