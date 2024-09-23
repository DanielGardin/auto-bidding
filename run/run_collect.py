from typing import Optional

from pathlib import Path
from time import perf_counter

import logging

import pandas as pd
from tqdm import tqdm

from bidding_train_env.import_utils import get_env, get_strategy, get_actor
from bidding_train_env.utils import get_root_path, turn_off_grad, set_seed
from bidding_train_env.envs import BiddingEnv
from bidding_train_env.strategy import (
    CollectStrategy,
    SigmaBiddingStrategy,
    BaseBiddingStrategy,
)
from data.rl_data import generate_rl_df


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def collect_rl_data(env: BiddingEnv, strategy: BaseBiddingStrategy, filename: str):
    save_path = get_root_path() / "data/traffic/rl_data"
    index = []
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for period in range(7, 28):
        try:
            env.set_period(period)
        except:
            continue

        for advertiser in tqdm(range(48)):
            env.advertiser_number = advertiser
            strategy.set_advertiser(advertiser)

            obs, info = env.reset()
            timeStepIndex = 0
            done = False

            while not done:
                bids = env.get_offline_bids()
                # action is retrievied from the offline bids values
                action = strategy.bid_to_action(bids, **obs)

                next_obs, reward, done, info = env.step(bids)

                states.append(strategy.preprocess(**obs).numpy().flatten())
                actions.append(action)
                rewards.append(reward)
                next_states.append(strategy.preprocess(**next_obs).numpy().flatten())
                dones.append(bool(done))

                obs = next_obs

                index.append(
                    {
                        "deliveryPeriodIndex": period,
                        "advertiserIndex": advertiser,
                        "timeStepIndex": timeStepIndex,
                    }
                )
                timeStepIndex += 1

    index = pd.MultiIndex.from_frame(pd.DataFrame(index))
    states = pd.DataFrame(states, index=index)
    actions = pd.DataFrame(actions, index=index)
    rewards = pd.DataFrame(rewards, index=index)
    next_states = pd.DataFrame(next_states, index=index)
    dones = pd.Series(dones, index=index)
    rl_data = generate_rl_df(states, actions, rewards, next_states, dones)
    rl_data.to_parquet(save_path / filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="SigmaBiddingStrategy")
    parser.add_argument("--filename", type=str, default="test_rl_data.parquet")
    args = parser.parse_args()
    strategy = CollectStrategy(
        base_strategy=get_strategy(
            args.strategy, actor="Actor"
        ),  # I know this is a string and not an Actor, but I wanted to just have a placeholder
    )

    env = get_env(
        "OfflineBiddingEnv",
        strategy,
        period=7,
    )

    collect_rl_data(env, strategy, args.filename)
