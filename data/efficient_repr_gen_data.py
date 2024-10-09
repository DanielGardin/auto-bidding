"""
    Store the data downloaded to a lighter format by converting it to a more efficient representations,
    with 3 different dataframes.
"""
from typing import Callable
from pathlib import Path
import logging

from bidding_train_env.utils import get_root_path

import pandas as pd
import tqdm

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()

        except Exception:
            self.handleError(record)


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(TqdmLoggingHandler())


DTYPES = {
    'deliveryPeriodIndex'    : int,
    'advertiserNumber'       : int,
    'advertiserCategoryIndex': int,
    'budget'                 : float,
    'CPAConstraint'          : float,
    'timeStepIndex'          : int,
    'remainingBudget'        : float,
    'pvIndex'                : int,
    'pValue'                 : float,
    'pValueSigma'            : float,
    'bid'                    : float,
    'xi'                     : bool,
    'adSlot'                 : int,
    'cost'                   : float,
    'isExposed'              : bool,
    'conversionAction'       : bool,
    'leastWinningCost'       : float,
    'isEnd'                  : bool
}


def process_advertiser_data(df: pd.DataFrame) -> pd.DataFrame:
    advertiser_data = df[[
        "advertiserNumber",
        "advertiserCategoryIndex",
        "budget",
        "CPAConstraint"
    ]].drop_duplicates()
    advertiser_data = advertiser_data.set_index('advertiserNumber')

    return advertiser_data


def process_bidding_data(df: pd.DataFrame) -> pd.DataFrame:
    bidding_data = df[[
        'timeStepIndex',
        'pvIndex',
        'advertiserNumber',
        'pValue',
        'pValueSigma',
        'bid'
    ]].copy()
    bidding_data = bidding_data.pivot(
        index   = ['timeStepIndex', 'pvIndex'],
        columns = ['advertiserNumber'],
        values  = ['pValue', 'pValueSigma', 'bid']
    )
    bidding_data = bidding_data.swaplevel(0, 1, axis=1)
    bidding_data = bidding_data.reindex(columns=sorted(bidding_data.columns, key=lambda x: x[0]))

    return bidding_data


def process_impression_data(df: pd.DataFrame) -> pd.DataFrame:
    impression_data = df[[
        'timeStepIndex',
        'pvIndex',
        'advertiserNumber',
        'xi',
        'adSlot',
        'cost',
        'isExposed',
        'conversionAction'
    ]].copy()
    impression_data = impression_data[impression_data['xi']].drop(columns=['xi'])

    # Set cost to zero, if not exposed
    # Daniel: I'm not sure if the cost is obfuscated when the ad is not exposed.
    # impression_data['cost'] = impression_data['cost'] * impression_data['isExposed']

    impression_data = impression_data.pivot(
        index=['timeStepIndex', 'pvIndex'],
        columns=['adSlot'],
        values=['advertiserNumber', 'cost', 'isExposed', 'conversionAction']
    )

    impression_data = impression_data.swaplevel(0, 1, axis=1)
    impression_data = impression_data.reindex(columns=sorted(impression_data.columns, key=lambda x: x[0]))

    return impression_data


def add_period_index(df: pd.DataFrame, period_number: int) -> pd.DataFrame:
    """
        Add the delivery period to the dataframe as an index at the top level.
    """
    new_df = df.copy()
    old_index_names = df.index.names

    new_df['deliveryPeriodIndex'] = period_number
    new_df.set_index('deliveryPeriodIndex', append=True, inplace=True)
    new_df = new_df.reorder_levels(['deliveryPeriodIndex', *old_index_names])

    return new_df


if __name__ == '__main__':
    data_path = get_root_path() / 'data/traffic/new_data_gen'

    save_dir = get_root_path() / 'data/traffic/data_gen_efficient_repr'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Sorting by version, so period-5.csv < period.12.csv
    version_fn: Callable[[Path], int] = lambda path: int(path.name.split("-")[-1].split(".")[0])

    csv_paths = sorted(
        data_path.glob("*.csv"),
        key=version_fn
    )

    impression_dfs = []
    for csv_path in tqdm.tqdm(csv_paths, desc='Processing period'):
        period_number = version_fn(csv_path)

        log.info("Reading period-%d.csv", period_number)
        df = pd.read_csv(csv_path).astype(DTYPES)

        log.info(f"Processing dataframes")

        # Get the first advertiser data. Assumes that the advertises are the same in all periods.
        if period_number == 7:
            advertiser_data = process_advertiser_data(df)
            advertiser_data.to_csv(save_dir / 'advertiser_data.csv')

        bidding_data    = process_bidding_data(df)
        bidding_data.to_parquet(save_dir / f'bidding-period-{period_number}.parquet')

        impression_data = process_impression_data(df)
        impression_data = add_period_index(impression_data, period_number)
        impression_dfs.append(impression_data)


    log.info("Done processing. Putting all periods together")

    impression_data = pd.concat(impression_dfs)
    # This generates a warning due to the columns being of different types.
    # Fix me: Solve this workaround.
    from io import BytesIO
    bytes_io = BytesIO()

    impression_data.to_parquet(bytes_io) # Warning!
    impression_data = pd.read_parquet(bytes_io)

    impression_data.columns = impression_data.columns.set_levels(impression_data.columns.levels[0].astype(int), level=0) # type: ignore
    impression_data.to_parquet(save_dir / 'impression_data.parquet')