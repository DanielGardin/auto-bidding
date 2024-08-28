from typing import Any, Optional

from pathlib import Path
from functools import partial

def get_root_path():
    return Path(__file__).parent.parent


def get_strategy(
        strategy_name: str,
        model_path: Optional[Path] = None,
        **strategy_kwargs
    ) -> partial[Any]:
    from . import strategy

    strategy_cls = getattr(strategy, strategy_name)

    try:
        return partial(strategy_cls, model_path=model_path, **strategy_kwargs)

    except TypeError:
        return partial(strategy_cls, **strategy_kwargs)