"""
    Download the challange data from the competition.
"""
from typing import Optional

from pathlib import Path
import logging
import os

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from tqdm import tqdm

logging.getLogger(__file__)

def download_data(period_name: str, target_dir: Path) -> None:
    periods = period_name.split('-')
    *first_periods, last_period = periods

    representation = f"period{'s ' + ', '.join(first_periods) + ' and' if len(periods) > 1 else ''} {last_period}"

    for period in periods:
        period_csv = target_dir / f'period-{period}.csv'

        if not period_csv.exists(): break


    # This else block will only run if the for loop above completes without breaking
    else:
        print(f"Data for {representation} already downloaded.")
        return


    url = f'https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_general_track_data_period_{period_name}.zip'

    response = urlopen(url)

    total_size = int(response.getheader('Content-Length'))

    chunk_size = 1024
    data = bytearray()

    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Downloading {representation}') as pbar:
        while len(data) < total_size:
            chunk = response.read(chunk_size)
            data.extend(chunk)
            pbar.update(len(chunk))
    
    buffer = BytesIO(data)

    with ZipFile(buffer) as zip_ref:
        print(f'Extracting {representation}...', end='\r')
        zip_ref.extractall(target_dir)
        print(f'Done extracting {representation}!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download data for the bidding competition.')
    parser.add_argument('--local', action='store_true', help='Download data into local directory.')

    args = parser.parse_args()

    if args.local:
        target_dir = Path(__file__).parent / 'traffic'

    elif os.path.exists('/hadatasets'):
        target_dir = Path('/hadatasets/auto-bidding')

    else:
        logging.info("Downloading outside of the cluster, using local directory by default.")
        target_dir = Path(__file__).parent / 'traffic'


    os.makedirs(target_dir, exist_ok=True)

    period_names = [
        '7-8',
        '9-10',
        '11-12',
        '13',
        '14-15',
        '16-17',
        '18-19',
        '20-21',
        '22-23',
        '24-25',
        '26-27',
    ]


    for period_name in period_names:
        download_data(period_name, target_dir)

    print('Done!')