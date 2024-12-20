from typing import Optional

from pathlib import Path
from time import perf_counter
from pathos.multiprocessing import ProcessingPool as Pool

import logging
import numpy as np
import pandas as pd
from torch import load
import torch
import pickle as pkl
import matplotlib.pyplot as plt

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

def get_sampled_advertiser_info(n : int, val_periods : list[int] = [25, 26, 27], seed : int = 42):
    np.random.seed(seed)
    advertiser_data = pd.concat([
        pd.read_csv(get_root_path() / f"data/traffic/{f}/advertiser_data.csv")
        for f in ["efficient_repr", "new_efficient_repr"]
    ])
    advertiser_number = advertiser_data.sample(n)[["advertiserNumber", "advertiserCategoryIndex"]].values
    advertiser_number, category = advertiser_number[:, 0], advertiser_number[:, 1]
    budget   = advertiser_data.sample(n)["budget"].values
    cpa      = advertiser_data.sample(n)["CPAConstraint"].values
    period   = np.random.choice(val_periods, n)
    return advertiser_number, budget, cpa, category, period


def evaluate_strategy_offline(
        env               : BiddingEnv,
        strategy          : BaseBiddingStrategy,
        advertiser_number : int,
        verbose           : bool = True
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
    env.advertiser_number = advertiser_number
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
        "score": info["score"],
        "cpa_r": strategy.cpa,
        "budget": strategy.budget,
        "category": strategy.category,
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
        "--val_periods",
        nargs="*",
        type=int,
        help="Periods to evaluate the strategy on.",
        default=[27]
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

    if hasattr(config.data, "state_norm") and config.data.state_norm:
        state_norm = pkl.load(open(get_root_path() / config.saved_models.state_norm, 'rb'))
    else:
        state_norm = None

    turn_off_grad(actor)


    if args.n_tests == 1:
        strategy = get_strategy(
            config.model.strategy,
            actor,
            config.model.budget,
            config.model.cpa,
            config.model.category,
            state_norm,
        )
        
        env = get_env(
            config.environment.environment,
            strategy,
            period=7,
            **config.environment.environment_params
        )

        evaluate_strategy_offline(env, strategy, True)
    else:
        torch.set_num_threads(1)
        advertiser_number, budget, cpa, category, period = get_sampled_advertiser_info(args.n_tests, args.val_periods)
        results = []

        aux = []
        for i in range(args.n_tests):
            strategy = get_strategy(
                config.model.strategy,
                actor,
                budget[i].item(),
                cpa[i].item(),
                category[i].item(),
                state_norm,
            )
            aux.append({
                "advertiser_number" : advertiser_number[i].item(), 
                "strategy" : strategy,
                "verbose" : False
                })

        env = get_env(
            "OfflineBiddingEnv",
            strategy = strategy,
            period = period[0].item(),
        )

        evaluate_strategy_offline_ = lambda kwargs: evaluate_strategy_offline(env, **kwargs)
        with Pool(args.n_jobs) as p:
            results = p.map(
                evaluate_strategy_offline_,
                aux
            )
           
        results = pd.DataFrame(results)

        # replace cpa inf with 50000
        results["cpa"] = results["cpa"].replace(np.inf, 50000)
        print(results.describe())


        
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].scatter(results["budget"], results["score"], alpha = 0.9)
        axs[0].set_xlabel("budget")
        axs[0].set_ylabel("score")
        axs[0].set_xscale("log")

        category_jitter = np.random.normal(0, 0.1, len(results)) + results["category"].values
        axs[1].scatter(category_jitter, results["score"], alpha = 0.9)
        axs[1].set_xlabel("category")
        axs[1].set_ylabel("score")

        axs[2].scatter(results["cpa_r"], results["score"], alpha = 0.9)
        axs[2].set_xlabel("cpa")
        axs[2].set_ylabel("score")
        axs[2].set_xscale("log")

        plt.suptitle(
            f"{args.config.split('/')[-2]} \nscore: {results.score.mean():.2f} +- {results.score.std():.2f}"
        )
        plt.tight_layout()
        
        
        plt.savefig(f"eval_{args.config.split('/')[-2]}.png")
