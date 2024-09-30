from typing import Optional

from pathlib import Path
from time import perf_counter

import logging

import numpy as np
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
    save_path = get_root_path() / "data/traffic/new_rl_data"
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
                # if any bid is nan, skip
                if np.isnan(bids).any():
                    break
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
    states.columns = strategy.state_names()
    actions = pd.DataFrame(actions, index=index)
    actions.columns = strategy.action_names()
    rewards = pd.DataFrame(rewards, index=index)
    rewards.columns = strategy.reward_names()
    next_states = pd.DataFrame(next_states, index=index)
    next_states.columns = strategy.state_names()
    dones = pd.Series(dones, index=index)
    rl_data = generate_rl_df(states, actions, rewards, next_states, dones)
    rl_data.to_parquet(save_path / filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="SigmaBiddingStrategy")
    parser.add_argument("--filename", type=str, default="updated_rl_data")
    args = parser.parse_args()

    df = []
    for data in ["old", "new"]:
        strategy = CollectStrategy(
            base_strategy=get_strategy(
                args.strategy, actor="Actor"
            ),  # I know this is a string and not an Actor, but I wanted to just have a placeholder
            data = data,
        )
        filename = args.filename + "_" + data + ".parquet"
        env = get_env(
            "OfflineBiddingEnv",
            strategy,
            data = data,
            period=7,
        )
        collect_rl_data(env, strategy, filename)
        df.append(pd.read_parquet(get_root_path() / "data/traffic/new_rl_data" / filename))

    df = pd.concat(df)
    filename = args.filename + ".parquet"
    df.to_parquet(get_root_path() / "data/traffic/new_rl_data" / filename)