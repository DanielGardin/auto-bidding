from omegaconf import OmegaConf
from torch import load
import pickle as pkl

from .base_bidding_strategy import BaseBiddingStrategy, BasePolicyStrategy
from .simple_strategy import SimpleBiddingStrategy as SimpleBiddingStrategy
from .sigma_strategy import SigmaBiddingStrategy as SigmaBiddingStrategy
from .alpha_strategy import AlphaBiddingStrategy as AlphaBiddingStrategy
from .alpha_punisher_strategy import AlphaPunisherStrategy as AlphaPunisherStrategy

from ..utils import get_root_path, turn_off_grad
from ..agents import actor

# Moved from `import_utils.py` due to circular import
def get_actor(actor_name: str, **kwargs) -> actor.Actor:
    return getattr(actor, actor_name)(**kwargs)


experiment_name = "latest"
if experiment_name == "latest": # little hack because I was forgetting to update the experiment name
    import os
    import pandas as pd
    experiment_names = os.listdir(get_root_path() / 'saved_models')
    dates = [name.split('_')[-1] for name in experiment_names]
    dates = pd.to_datetime(dates, errors = "coerce")
    experiment_name = experiment_names[dates.argmax()]

    print(f"Latest experiment: {experiment_name}")


config_path = get_root_path() / f'saved_models/{experiment_name}/config.yaml'
try:
    config = OmegaConf.load(config_path)
except:
    pass

strategy = globals()[config.model.strategy]

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
        agent.load_state_dict(load(model_path, map_location='cpu'))

        turn_off_grad(agent)

        if hasattr(config.data, "state_norm") and config.data.state_norm:
            state_norm = pkl.load(open(get_root_path() / config.saved_models.state_norm, 'rb'))
        else:
            state_norm = None
        super().__init__(
            agent,
            budget,
            name,
            cpa,
            category,
            state_norm
        )