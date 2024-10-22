import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl

from bidding_train_env.import_utils import get_env, get_strategy, get_actor
from bidding_train_env.utils import get_root_path
from data.rl_data import generate_rl_df
from copy import deepcopy


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


dummy_actor = get_actor("DeterministicMLP", input_dim=10, hidden_dims=[10], output_dim=1)
dummy_strategy = get_strategy("AlphaBiddingStrategy", actor=dummy_actor, budget=1e6, cpa=1e6, category=0)

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


def save_history(
    data          : str,
):
    save_path = get_root_path() / f"data/traffic/history/{data}"

    for period in tqdm(range(7, 28)):
        history = {}
        env = get_env(
            "OfflineBiddingEnv",
            dummy_strategy,
            data = data,
            period = period,
            advertiser_number = 0,
        )
        
        for advertiser in range(48):
            budget, cpa, category = get_advertiser_info(advertiser, data)
            strategy = get_strategy(
                "AlphaBiddingStrategy",
                actor    = dummy_actor,
                budget   = budget,
                cpa      = cpa,
                category = category,
            )

            env.set_strategy(strategy)
            env.advertiser_number = advertiser

            _, _ = env.reset()
            for _ in range(48):
                next_obs, _, _, _ = env.step()

            history[advertiser] = deepcopy(next_obs)

        with open(save_path / f"period_{period}.pkl", "wb") as f:
            pkl.dump(history, f)


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

    for period in tqdm(range(7, 28)):
        history = pkl.load(open(get_root_path() / f"data/traffic/history/{data}/period_{period}.pkl", "rb"))

        for advertiser in range(48):
            h_ad = history[advertiser]
            budget, cpa, category = get_advertiser_info(advertiser, data)

            strategy = get_strategy(
                strategy_name,
                actor    = dummy_actor,
                budget   = budget,
                cpa      = cpa,
                category = category,
            )
            
            obs = {
                "timeStepIndex" : 0,
                "pValues" : h_ad["historyPValueInfo"][0][:, 0],
                "pValueSigmas" : h_ad["historyPValueInfo"][0][:, 1],
                "historyPValueInfo" : [],
                "historyBid" : [],
                "historyAuctionResult"   : [],
                "historyImpressionResult": [],
                "historyLeastWinningCost": [],
            }
            for timeStepIndex in range(48):
                states.append(strategy.preprocess(**obs).numpy().flatten())

                bids = h_ad["historyBid"][timeStepIndex]
                cost = h_ad["historyAuctionResult"][timeStepIndex][:, 2].sum()
                strategy.pay(cost)

                action = strategy.bid_to_action(bids, **obs)

                next_obs = {
                    "timeStepIndex" : timeStepIndex + 1,
                    "pValues" : h_ad["historyPValueInfo"][(timeStepIndex + 1)][:, 0] if timeStepIndex < 47 else np.array([]),
                "pValueSigmas" : h_ad["historyPValueInfo"][(timeStepIndex + 1)][:, 1] if timeStepIndex < 47 else np.array([]),
                    "historyPValueInfo" : h_ad["historyPValueInfo"][:timeStepIndex + 1],
                    "historyBid" : h_ad["historyBid"][:timeStepIndex + 1],
                    "historyAuctionResult"   : h_ad["historyAuctionResult"][:timeStepIndex + 1],
                    "historyImpressionResult": h_ad["historyImpressionResult"][:timeStepIndex + 1],
                    "historyLeastWinningCost": h_ad["historyLeastWinningCost"][:timeStepIndex + 1],
                }
                reward = strategy.get_reward(**obs)
                done = False if timeStepIndex < 47 else True
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
    states.columns = strategy.state_names
    actions = pd.DataFrame(actions, index=index)
    actions.columns = strategy.action_names
    rewards = pd.DataFrame(rewards, index=index)
    rewards.columns = strategy.reward_names
    next_states = pd.DataFrame(next_states, index=index)
    next_states.columns = strategy.state_names
    dones = pd.Series(dones, index=index)
    rl_data = generate_rl_df(states, actions, rewards, next_states, dones)
    rl_data.to_parquet(save_path / filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_history", type=bool, default=False)
    parser.add_argument("--strategy", type=str, default="AlphaBiddingStrategy")
    parser.add_argument("--filename", type=str, default="updated_rl_data")
    args = parser.parse_args()

    if args.save_history:
        for data in ["old", "new"]:
            filename = "history_" + data + ".pkl"
            save_history(data)

    df = []
    for data in ["old", "new"]:
        filename = args.filename + "_" + data + ".parquet"
        collect_rl_data(args.strategy, filename, data)
        df.append(pd.read_parquet(get_root_path() / "data/traffic/new_rl_data" / filename))

    filename = args.filename + ".parquet"

    df = pd.concat(df)
    state_norm = {}
    state_norm["mean"] = df["state"].mean().values
    state_norm["std"] = df["state"].std().values
    filename_pkl = filename.replace(".parquet", "_norm.pkl")
    with open(get_root_path() / "data/traffic/new_rl_data" / filename_pkl, "wb") as f:
        pkl.dump(state_norm, f)

    df.to_parquet(get_root_path() / "data/traffic/new_rl_data" / filename)