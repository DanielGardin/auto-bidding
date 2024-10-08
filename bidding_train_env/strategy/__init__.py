from omegaconf import OmegaConf
from torch import load

from .base_bidding_strategy import BaseBiddingStrategy, BasePolicyStrategy
from .simple_strategy import SimpleBiddingStrategy as SimpleBiddingStrategy
from .sigma_strategy import SigmaBiddingStrategy as SigmaBiddingStrategy
from .alpha_strategy import AlphaBiddingStrategy as AlphaBiddingStrategy

from ..utils import get_root_path, turn_off_grad
from ..agents import actor

# Moved from `import_utils.py` due to circular import
def get_actor(actor_name: str, **kwargs) -> actor.Actor:
    return getattr(actor, actor_name)(**kwargs)


experiment_name = "iql_alpha_2024-10-01-12:43:56"

config_path = get_root_path() / f'saved_models/{experiment_name}/config.yaml'
strategy    = AlphaBiddingStrategy
try:
    config = OmegaConf.load(config_path)

except:
    pass

class PlayerBiddingStrategy(strategy):
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

        agent = get_actor(config.model.actor, **config.model.actor_params)

        model_path = get_root_path() / config.saved_models.actor
        print(model_path)
        agent.load_state_dict(load(model_path, map_location='cpu'))

        turn_off_grad(agent)


        super().__init__(
            agent,
            budget,
            name,
            cpa,
            category
        )