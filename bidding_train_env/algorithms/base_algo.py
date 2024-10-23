from typing import Any, Optional, Mapping

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from tensordict import TensorDict
from torch.optim.lr_scheduler import LRScheduler

from ..utils import get_root_path
from ..replaybuffer import AbstractReplayBuffer
from ..envs import BiddingEnv

from collections import defaultdict

import datetime
import tqdm

from torch.utils.tensorboard import SummaryWriter # type: ignore
import wandb

from abc import ABC, abstractmethod

import logging
# import os
# os.environ['WANDB_DIR'] = os.getcwd()


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)


class DummyWriter(SummaryWriter):
    def __init__(*args, **kwargs):
        pass

    def __getattribute__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


class EpochLogs:
    def __init__(self):
        self.logs: dict[str, list] = {}

    def log(self, log_info: Mapping[str, Any]):
        for key, value in log_info.items():
            if key not in self.logs:
                self.logs[key] = []

            self.logs[key].append(value)

    def mean_log(self) -> dict[str, Any]:
        return {
            key: np.nanmean(values) for key, values in self.logs.items()
        }

    def std_log(self) -> dict[str, Any]:
        return {
            key: np.nanstd(values) for key, values in self.logs.items()
        }

    def clear(self):
        self.logs.clear()


class RLAlgorithm(nn.Module, ABC):
    @abstractmethod
    def train_step(self, batch: TensorDict) -> dict[str, Any]:
        pass

    
    # These methods are for future use
    def on_epoch_start(self, env: BiddingEnv, replay_buffer: AbstractReplayBuffer) -> dict[str, Any]:
        return {}


    def on_epoch_end(self) -> dict[str, Any]:
        return {}


    def begin_experiment(
            self,
            project_name: str,
            experiment_name: str,
            log_dir: str | Path,
            checkpoint_interval: Optional[int] = None,
            use_wandb: bool = False,
            config: Optional[dict] = None
        ):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        if "_date" in experiment_name:
            self.experiment_name = f"{experiment_name.replace('_date', '')}_{timestamp}"
        else:
            self.experiment_name = experiment_name

        log_dir = Path(log_dir)

        self.checkpoint_interval = checkpoint_interval

        if checkpoint_interval is not None:
            self.checkpoint_dir = get_root_path() / f"checkpoints/{experiment_name}"
        

        #log_dir.mkdir(parents=True, exist_ok=True)
        if use_wandb:
            self.run = wandb.init(
                project          = project_name,
                name             = experiment_name,
                config           = config,
                dir              = log_dir,
                sync_tensorboard = True
            )

        self.writer = SummaryWriter(log_dir / 'tensorboard')
        self.config = config

        self.global_step = 0


    def _write_logs(self, tag: str, logs: dict[str, Any]):
        for key, value in logs.items():
            self.writer.add_scalar(f"{tag}/{key}", value, self.global_step)


    def learn(
            self,
            num_epochs: int,
            steps_per_epoch: int,
            replay_buffer: AbstractReplayBuffer,
            batch_size: int,
            env: BiddingEnv,
            lr_scheduler: Optional[LRScheduler] = None,
            val_periods: Optional[list[int]] = None,
        ):
        running_experiment = hasattr(self, "writer")
        if not running_experiment:
            logger.info("Calling learn before begin_experiment, no logging will be done and no weights will be saved.")

            self.checkpoint_interval = None
            self.writer              = DummyWriter()

            self.global_step = 0

        if val_periods is None:
            val_periods = []

        epoch_logs = EpochLogs()
        eval_logs  = EpochLogs()
        for epoch in range(1, num_epochs+1):
            self.train()

            with tqdm.tqdm(total=steps_per_epoch, unit=" steps", dynamic_ncols=True, ncols=300) as pbar:

                pbar.set_description(f"Epoch {epoch}/{num_epochs}")

                start_logs = self.on_epoch_start(env, replay_buffer)
                self._write_logs("train", start_logs)

                epoch_logs.clear()
                for step in range(steps_per_epoch):
                    batch = replay_buffer.sample(batch_size)
                    train_logs = self.train_step(batch)

                    epoch_logs.log(train_logs)
                    pbar.set_postfix(train_logs)
                    pbar.update()

                    self.global_step += 1
                
                epoch_summary = epoch_logs.mean_log()
                pbar.set_postfix(epoch_summary)
                self._write_logs("train", epoch_summary)
                
                end_logs = self.on_epoch_end()
                self._write_logs("train", end_logs)

                pbar.close()

            eval_logs.clear()
            for period in val_periods:
                eval_log = self.evaluate(env, period)

                eval_logs.log(eval_log)

            eval_summary = eval_logs.mean_log()
            self._write_logs("eval", eval_summary)

            if self.checkpoint_interval is not None and epoch % self.checkpoint_interval == 0:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

                for net_name, state_dict in self.save().items():
                    torch.save(state_dict, self.checkpoint_dir / f"{net_name}_checkpoint_{epoch}.pth")

            if lr_scheduler is not None:
                lr_scheduler.step()


        if running_experiment:
            self.writer.close()

            if hasattr(self, "run"):
                self.run.finish()

            save_dir = get_root_path() / "saved_models" / self.experiment_name
            save_dir.mkdir(parents=True, exist_ok=True)

            save_dict = self.save()
            for net_name, state_dict in save_dict.items():
                torch.save(state_dict, save_dir / f"{net_name}.pth")

            if hasattr(self, "config") and self.config is not None:
                import yaml

                self.config["saved_models"] = {
                    net_name : str((save_dir / f"{net_name}.pth").relative_to(get_root_path())) for net_name in save_dict
                }

                with open(save_dir / 'config.yaml', 'w') as f:
                    yaml.dump(self.config, f)


    def save(self) -> dict[str, Any]:
        return {}


    def evaluate(
            self,
            env: BiddingEnv,
            period: int
        ) -> dict[str, Any]:
        self.eval()

        env.set_period(period)
        obs, info = env.reset()

        cum_reward = 0
        done = False
        with torch.no_grad():
            while not done:
                bids = env.strategy.bidding(**obs)

                obs, reward, done, info = env.step(bids)

                cum_reward += reward

        info[f"total_reward"] = cum_reward

        env.strategy.reset()

        return info
