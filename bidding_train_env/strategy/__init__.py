from typing import Optional, Protocol

from pathlib import Path
from functools import partial

from .base_bidding_strategy import BaseBiddingStrategy
from .player_bidding_strategy import PlayerBiddingStrategy
from .onlinelp_bidding_strategy import OnlineLpBiddingStrategy
from .simple_strategy import SimpleBiddingStrategy
from .dt_bidding_strategy import DtBiddingStrategy


class BiddingAgentConstructor(Protocol):
    def __call__(
            self,
            budget: float = 100.,
            name: str = "BaseStrategy",
            cpa: float = 2.,
            category: int = 1
        ) -> BaseBiddingStrategy: ...


def get_strategy(
        strategy_name: str,
        model_path: Optional[Path] = None,
        **strategy_kwargs
    ) -> BiddingAgentConstructor:

    for strategy_file in Path(__file__).parent.rglob("*strategy.py"):
        module = __import__(f"bidding_train_env.strategy.{strategy_file.stem}", fromlist=[""])

        if strategy_name in dir(module):
            strategy_cls = getattr(module, strategy_name)
            break
    else:
        raise ValueError(f"Strategy {strategy_name} not found.")


    def strategy_constructor(
            *strategy_args,
            **strategy_kwargs
        ) -> BaseBiddingStrategy:
        try:
            return strategy_cls(*strategy_args, model_path=model_path, **strategy_kwargs)
        
        except TypeError:
            return strategy_cls(*strategy_args, **strategy_kwargs)
        
    return strategy_constructor