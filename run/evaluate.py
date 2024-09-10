from typing import Optional

from pathlib import Path
from time import perf_counter

import logging
import numpy as np

from bidding_train_env.import_utils import get_env, get_strategy
from bidding_train_env.envs import BiddingEnv
from bidding_train_env.strategy import BaseBiddingStrategy

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_strategy_offline(
        env: BiddingEnv,
        strategy: BaseBiddingStrategy
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
        "strategy",
        type=str,
        help="Bidding strategy to evaluate."
    )
    parser.add_argument(
        "--path",
        '-p',
        type=str,
        default=None,
        help="Path to the model to load."
    )

    args = parser.parse_args()

    model_path = None
    if args.path is not None:
        model_path = Path(args.path)

    strategy = get_strategy(args.strategy, model_path)()

    evaluate_strategy_offline(agent)