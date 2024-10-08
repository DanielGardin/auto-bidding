import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from bidding_train_env.import_utils import get_env, get_strategy, get_actor
from bidding_train_env.utils import get_root_path
from data.rl_data import generate_rl_df


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


dummy_actor = get_actor("DeterministicMLP", input_dim=10, hidden_dims=[10], output_dim=1)

def get_advertiser_info(
    advertiser : int, 
    data       : str
):  
    
    if data == "new":
        advertiser_data = pd.read_csv(
            get_root_path() / "data/traffic/new_efficient_repr/advertiser_data.csv", index_col=0
        )
    else:
        advertiser_data = pd.read_csv(
            get_root_path() / "data/traffic/efficient_repr/advertiser_data.csv", index_col=0
        )

    budget   = advertiser_data.loc[advertiser, "budget"]
    cpa      = advertiser_data.loc[advertiser, "CPAConstraint"]
    category = advertiser_data.loc[advertiser, "advertiserCategoryIndex"]
    return budget, cpa, category


def collect_rl_data(
    strategy_name : str, 
    filename      : str,
    data          : str,
):
    save_path = get_root_path() / "data/traffic/new_rl_data"
    index       = []
    states      = []
    actions     = []
    rewards     = []
    next_states = []
    dones       = []

    for period in range(7, 28):
        for advertiser in tqdm(range(48)):
            
            budget, cpa, category = get_advertiser_info(advertiser, data)

            strategy = get_strategy(
                strategy_name,
                actor    = dummy_actor,
                budget   = budget,
                cpa      = cpa,
                category = category,
            )

            env = get_env(
                "OfflineBiddingEnv",
                strategy,
                data = data,
                period = period,
                advertiser_number = advertiser,
            )

            obs, info = env.reset()

            for timeStepIndex in range(48):
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
                        "advertiserNumber": advertiser,
                        "timeStepIndex": timeStepIndex,
                    }
                )

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
    parser.add_argument("--strategy", type=str, default="AlphaBiddingStrategy")
    parser.add_argument("--filename", type=str, default="updated_rl_data")
    args = parser.parse_args()

    df = []
    for data in ["old", "new"]:
        filename = args.filename + "_" + data + ".parquet"
        collect_rl_data(args.strategy, filename, data)
        df.append(pd.read_parquet(get_root_path() / "data/traffic/new_rl_data" / filename))

    df = pd.concat(df)
    filename = args.filename + ".parquet"
    df.to_parquet(get_root_path() / "data/traffic/new_rl_data" / filename)