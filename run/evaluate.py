from typing import Optional

from pathlib import Path
from time import perf_counter

import logging

from torch import load

from bidding_train_env.import_utils import get_env, get_strategy, get_actor
from bidding_train_env.utils import get_root_path, turn_off_grad, set_seed
from bidding_train_env.envs import BiddingEnv
from bidding_train_env.strategy import BaseBiddingStrategy

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_strategy_offline(
        env: BiddingEnv,
        strategy: BaseBiddingStrategy,
        period: int = 7
    ):
    """
    offline evaluation
    """
    start = perf_counter()

    logger.info(
        f"Evaluating strategy {strategy} on environment {env}."
    )

    cumulative_reward = 0
    num_steps = 0

    env.set_period(period)
    obs, info = env.reset()

    done = False
    while not done:
        action = strategy.bidding(**obs)
        obs, reward, done, info = env.step(action)

        cumulative_reward += reward
        num_steps += 1
    
    total_time = perf_counter() - start

    if total_time < 1:
        time_str = f"{int(1000*total_time)} miliseconds"
    
    elif total_time < 60:
        time_str = f"{total_time:.3f} seconds"

    else:
        time_str = f"{total_time//60} min {int(total_time)%60} seconds"

    logger.info(f"Strategy evaluation completed within {num_steps} steps, in {time_str}.")
    logger.info(f"Total reward: {cumulative_reward}")
    logger.info(f"Total cost: {strategy.budget - env.remaining_budget}")
    logger.info(f"Total conversions: {info['conversions']}")
    logger.info(f"Total wins: {info['wins']}")
    logger.info(f"CPA: {info['cpa']}")
    logger.info(f"Score: {info['score']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Path to yaml config file.",
        nargs="?",
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    set_seed(config.general.seed)

    actor = get_actor(config.model.actor, **config.model.actor_params)
    actor = actor.to(config.general.device)

    model_path = get_root_path() / config.saved_models.actor

    state_dict = load(model_path, map_location=config.general.device)
    actor.load_state_dict(state_dict)

    turn_off_grad(actor)

    strategy = get_strategy(
        config.model.strategy,
        actor,
        config.model.budget,
        config.model.cpa,
        config.model.category
    )

    env = get_env(
        config.environment.environment,
        strategy,
        period=7,
        **config.environment.environment_params
    )

    evaluate_strategy_offline(env, strategy, period=7)