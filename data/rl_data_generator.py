from typing import Callable
from numpy.typing import NDArray

from pathlib import Path
import pandas as pd
import numpy as np

from bidding_train_env.utils import get_root_path
from rl_data import generate_rl_df

import tqdm

history_features = [
    "remaining_budget",
    "historical_mean_bid",
    "last_three_bid_mean",
    "historical_mean_least_winning_cost",
    "last_three_least_winning_cost_mean",
    "historical_mean_pValues",
    "last_three_pValues_mean",
    "historical_conversion_mean",
    "last_three_conversion_mean",
    "historical_xi_mean",
    "last_three_xi_mean",
    "historical_pv_num_total",
    "last_three_pv_num_total",
]

current_features = [
    "time_left",
    "current_pValues_mean",
    "current_pv_num",
]

state_features = [
    "time_left",
    "remaining_budget",
    "historical_mean_bid",
    "last_three_bid_mean",
    "historical_mean_least_winning_cost",
    "historical_mean_pValues",
    "historical_conversion_mean",
    "historical_xi_mean",
    "last_three_least_winning_cost_mean",
    "last_three_pValues_mean",
    "last_three_conversion_mean",
    "last_three_xi_mean",
    "current_pValues_mean",
    "current_pv_num",
    "last_three_pv_num_total",
    "historical_pv_num_total"
]

def generate_rl_data(data_path: Path, min_budget: float = 0.1):
    # Sorting by version, so period-5.csv < period.12.csv
    version_fn: Callable[[Path], int] = lambda path: int(path.name.split("-")[-1].split(".")[0])

    # Loading the data

    advertiser_metadata = pd.read_csv(
        data_path / 'advertiser_data.csv',
        index_col='advertiserNumber'
    )

    impression_data = pd.read_parquet(data_path / f'impression_data.parquet')


    least_winning_costs = impression_data[(3, 'cost')] \
        .groupby(['deliveryPeriodIndex', 'timeStepIndex']).mean()

    impression_data[(1, 'cost')] = impression_data[(1, 'cost')] * impression_data[(1, 'isExposed')]
    impression_data[(2, 'cost')] = impression_data[(2, 'cost')] * impression_data[(2, 'isExposed')]
    impression_data[(3, 'cost')] = impression_data[(3, 'cost')] * impression_data[(3, 'isExposed')]

    winning_pValues = pd.DataFrame(
        index=impression_data.index,
        columns=pd.MultiIndex.from_tuples([
            (col, "pValue") for col in impression_data.columns.get_level_values(0).unique()
        ])
    )
    period_paths = sorted(
        data_path.glob("*[0-9].parquet"),
        key=version_fn
    )

    periods = [version_fn(path) for path in period_paths]


    # Prepating dataframes

    index = pd.MultiIndex.from_product(
        [periods, range(48), range(48)],
        names=['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex']
    )

    history_state = pd.DataFrame(index=index, columns=history_features)
    current_state = pd.DataFrame(index=index, columns=current_features)
    actions       = pd.Series(index=index)

    # Processing bidding data

    advertiser_indices = impression_data \
        .loc[:, impression_data.columns.get_level_values(1) == 'advertiserNumber']

    for period_number, period_path in tqdm.tqdm(
        zip(periods, period_paths), total=len(periods), desc='Processing period'
        ):
        period_data = pd.read_parquet(period_path)

        # Selecting pValues to the impression data
        pValues = period_data.loc[:, period_data.columns.get_level_values(1) == 'pValue']

        winning_pValues.loc[period_number] = pValues.to_numpy()[
            np.arange(len(pValues))[:, None],                     # Takes one value for each row
            advertiser_indices.loc[period_number].to_numpy()      # selecting the corresponding advertiserNumber column
        ]

        period_data = period_data.stack(0, future_stack=True) \
            .reorder_levels(['advertiserNumber', 'timeStepIndex', 'pvIndex']) \
            .sort_index(level=0).groupby(['advertiserNumber', 'timeStepIndex']).mean()

        current_state.loc[period_number, "current_pValues_mean"] = period_data['pValue'].to_numpy()

        historical_means = period_data.groupby('advertiserNumber') \
            .expanding().mean().droplevel(0)

        last_three_means = period_data.groupby('advertiserNumber') \
            .rolling(3, min_periods=1).mean().droplevel(0)

        history_state.loc[period_number, "historical_mean_bid"] = historical_means['bid'].to_numpy()
        history_state.loc[period_number, "last_three_bid_mean"] = last_three_means['bid'].to_numpy()

        history_state.loc[period_number, "historical_mean_pValues"] = historical_means['pValue'].to_numpy()
        history_state.loc[period_number, "last_three_pValues_mean"] = last_three_means['pValue'].to_numpy()

        # If, for some reason, the pValues at a timestep are all 0, we set the action to 0
        actions.loc[(period_number,)] = (period_data['bid'] / period_data['pValue']).fillna(0.).to_numpy()


    # Processing impression data

    impression_data = pd.concat([impression_data, winning_pValues], axis=1) \
        .sort_index(axis=1, level=0)

    win_results = impression_data.stack(0, future_stack=True)
    win_results['xi'] = 1
    win_results['pValue'] *= win_results['isExposed']

    win_results = win_results \
        .groupby(['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex']) \
        .sum().reindex(index, fill_value=0)

    pv_num  = impression_data \
        .groupby(['deliveryPeriodIndex', 'timeStepIndex']).size()
    
    # Expand pv_num to all advertisers
    new_index = pd.MultiIndex.from_product([
            pv_num.index.get_level_values(0).unique(),
            pv_num.index.get_level_values(1).unique(),
            advertiser_metadata.index.get_level_values(0)
    ], names=pv_num.index.names + advertiser_metadata.index.names)

    pv_num = pv_num.reindex(new_index) \
        .swaplevel("advertiserNumber", "timeStepIndex").sort_index()
    
    least_winning_costs = least_winning_costs.reindex(new_index) \
        .swaplevel("advertiserNumber", "timeStepIndex").sort_index()

    conversion_per_pv = win_results['conversionAction'] \
        .div(pv_num.to_numpy()).groupby(['deliveryPeriodIndex', 'advertiserNumber'])

    win_count = win_results['xi'] \
        .div(pv_num.to_numpy()).groupby(['deliveryPeriodIndex', 'advertiserNumber'])

    cumulative_cost = win_results['cost'] \
        .groupby(['deliveryPeriodIndex', 'advertiserNumber']).cumsum()

    budgets = advertiser_metadata['budget']

    # Building the state

    current_state['time_left'] = 1 - index.get_level_values('timeStepIndex') / 48

    current_state["current_pv_num"] = pv_num.to_numpy() / (500_000 / 48)

    history_state['remaining_budget'] = (budgets - cumulative_cost) / budgets

    history_state['historical_mean_least_winning_cost'] = least_winning_costs \
        .groupby(['deliveryPeriodIndex', 'advertiserNumber']) \
        .expanding().mean().to_numpy()

    history_state['last_three_least_winning_cost_mean'] = least_winning_costs \
        .groupby(['deliveryPeriodIndex', 'advertiserNumber']) \
        .rolling(3, min_periods=1).mean().to_numpy()

    history_state["historical_pv_num_total"] = pv_num \
        .groupby(['deliveryPeriodIndex', 'advertiserNumber']) \
        .cumsum().to_numpy() / 500_000

    history_state["last_three_pv_num_total"] = pv_num \
        .groupby(['deliveryPeriodIndex', 'advertiserNumber']) \
        .rolling(3, min_periods=1).sum().to_numpy() / (500_000 * 3 / 48)


    history_state['historical_conversion_mean'] = conversion_per_pv \
        .expanding().mean().to_numpy()

    history_state['last_three_conversion_mean'] = conversion_per_pv \
        .rolling(3, min_periods=1).mean().to_numpy()


    history_state['historical_xi_mean'] = win_count \
        .expanding().mean().to_numpy()

    history_state['last_three_xi_mean'] = win_count \
        .rolling(3, min_periods=1).mean().to_numpy()


    # Collecting rl data

    state      = pd.DataFrame(index=index, columns=state_features)
    next_state = pd.DataFrame(index=index, columns=state_features)
    rewards    = pd.DataFrame(index=index, columns=['discrete', 'continuous'])

    state[current_features] = current_state
    state[history_features] = history_state \
        .groupby(['deliveryPeriodIndex', 'advertiserNumber']).shift(1, fill_value=0)

    state.loc[(slice(None), slice(None), 0), "remaining_budget"] = 1.

    next_state[current_features] = current_state \
        .groupby(['deliveryPeriodIndex', 'advertiserNumber']).shift(-1, fill_value=0)
    next_state[history_features] = history_state

    rewards['discrete']   = win_results['conversionAction']
    rewards['continuous'] = win_results['pValue']

    dones = (state["remaining_budget"] < min_budget) | (index.get_level_values('timeStepIndex') == 47)

    return state, actions, rewards, next_state, dones


if __name__ == "__main__":
    data_path = get_root_path() / 'data/traffic/efficient_repr'
    save_path = get_root_path() / 'data/traffic/rl_data'

    states, actions, rewards, next_states, dones = generate_rl_data(
        data_path,
        min_budget = 0.1
    )

    rl_data = generate_rl_df(
        states,
        actions,
        rewards,
        next_states,
        dones)
    
    rl_data.to_parquet(save_path / 'rl_data.parquet')