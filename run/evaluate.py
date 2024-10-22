from typing import Optional

from pathlib import Path
from time import perf_counter
from pathos.multiprocessing import ProcessingPool as Pool

import logging
import numpy as np
import pandas as pd
from torch import load
import torch

from bidding_train_env.import_utils import get_env, get_strategy, get_actor
from bidding_train_env.utils import get_root_path, turn_off_grad, set_seed
from bidding_train_env.envs import BiddingEnv
from bidding_train_env.strategy import BaseBiddingStrategy
from bidding_train_env.agents import Actor

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def get_sampled_advertiser_info(n : int):
    advertiser_data = pd.concat([
        pd.read_csv(get_root_path() / f"data/traffic/{f}/advertiser_data.csv")
        for f in ["efficient_repr", "new_efficient_repr"]
    ])
    budget   = advertiser_data.sample(n)["budget"].values
    cpa      = advertiser_data.sample(n)["CPAConstraint"].values
    category = advertiser_data.sample(n)["advertiserCategoryIndex"].values
    period   = np.random.choice([13], n)
    return budget, cpa, category, period


def evaluate_strategy_offline(
        env      : BiddingEnv,
        strategy : BaseBiddingStrategy,
        verbose  : bool = True
    ):
    """
    offline evaluation
    """

    start = perf_counter()

    if verbose:
        logger.info(
            f"Evaluating strategy {strategy} on environment {env}."
        )

    cumulative_reward = 0
    num_steps = 0

    env.strategy = strategy
    env.reset()
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

    if verbose:
        logger.info(f"Strategy evaluation completed within {num_steps} steps, in {time_str}.")
        logger.info(f"Total reward: {cumulative_reward}")
        logger.info(f"Total cost: {strategy.budget - env.remaining_budget}")
        logger.info(f"Total conversions: {info['conversions']}")
        logger.info(f"Total wins: {info['wins']}")
        logger.info(f"CPA: {info['cpa']}")
        logger.info(f"Score: {info['score']}")

    return {
        "cumulative_reward": cumulative_reward,
        "cost" : strategy.budget - env.remaining_budget,
        "conversions": info["conversions"],
        "cpa" : info["cpa"],
        "score": info["score"]
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Path to yaml config file.",
        nargs="?",
    )
    parser.add_argument(
        "--n_tests",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
    )
    device = "cpu"
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    set_seed(config.general.seed)

    actor = get_actor(config.model.actor, **config.model.actor_params)
    actor = actor.to(device)

    model_path = get_root_path() / config.saved_models.actor

    state_dict = load(model_path, map_location=device)
    actor.load_state_dict(state_dict)

    turn_off_grad(actor)


    if args.n_tests == 1:
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
    else:
        torch.set_num_threads(1)
        budget, cpa, category, period = get_sampled_advertiser_info(args.n_tests)
        results = []

        aux = []
        for i in range(args.n_tests):
            strategy = get_strategy(
                config.model.strategy,
                actor,
                budget[i].item(),
                cpa[i].item(),
                category[i].item()
            )
            aux.append(strategy)

        env = get_env(
            "OfflineBiddingEnv",
            strategy = strategy,
            period = period[0].item(),
        )

        evaluate_strategy_offline_ = lambda strategy: evaluate_strategy_offline(env, strategy, False)
        with Pool(args.n_jobs) as p:
            results = p.map(
                evaluate_strategy_offline_,
                aux
            )
           
        results = pd.DataFrame(results)

        # replace cpa inf with 50000
        results["cpa"] = results["cpa"].replace(np.inf, 50000)
        print(results.describe())
