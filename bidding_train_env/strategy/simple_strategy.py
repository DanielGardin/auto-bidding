from typing import Optional
from numpy.typing import NDArray

from pathlib import Path
import numpy as np

import torch

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from ..envs.state_preprocessing import statistics_observation
from ..utils import get_root_path


class SimpleBiddingStrategy(BaseBiddingStrategy):
    """
    Behavioral Cloning (bc) Strategy
    """

    def __init__(
            self,
            model_path: Optional[Path] = None,
            budget=100,
            name="SimpleStrategy",
            cpa=2,
            category=1
        ):
        super().__init__(budget, name, cpa, category)

        if model_path is None:
            model_path = get_root_path() / 'saved_model' / 'BCtest' / 'bc_model.pth'

        self.model = torch.jit.load(model_path)


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

        state = statistics_observation(
            self.remaining_budget,
            timeStepIndex,
            pValues,
            pValueSigmas,
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
        )

        test_state = torch.tensor(state, dtype=torch.float)
        alpha = self.model(test_state)
        alpha = alpha.cpu().numpy()
        bids = alpha * pValues

        return bids
