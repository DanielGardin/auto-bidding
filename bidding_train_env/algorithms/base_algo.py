from typing import Any, Optional

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


class RLAlgorithm(nn.Module, ABC):
    @abstractmethod
    def train_step(self, batch: TensorDict) -> dict[str, Any]:
        pass

    
    # These methods are for future use
    def on_epoch_start(self, replay_buffer: AbstractReplayBuffer) -> dict[str, Any]:
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
        self.experiment_name = f"{experiment_name}_{timestamp}"

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


    def learn(
            self,
            num_epochs: int,
            steps_per_epoch: int,
            replay_buffer: AbstractReplayBuffer,
            batch_size: int,
            eval_env: BiddingEnv,
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

        def write_info(tag: str, info: dict[str, Any]):
            for key, value in info.items():
                self.writer.add_scalar(f"{tag}/{key}", value, self.global_step)

        epoch_info = defaultdict(list)
        eval_total = defaultdict(list)
        for epoch in range(1, num_epochs + 1):
            self.train()

            pbar = tqdm.tqdm(
                range(steps_per_epoch),
                desc=f"Epoch {epoch}/{num_epochs}"
            )

            start_info = self.on_epoch_start(replay_buffer)
            write_info("train", start_info)

            epoch_info.clear()
            for step in pbar:
                batch = replay_buffer.sample(batch_size)

                train_info = self.train_step(batch)

                pbar.set_postfix(train_info)

                for key, value in train_info.items():
                    epoch_info[key].append(value)

                self.global_step += 1

            for key, value in epoch_info.items():
                mean_value = np.nanmean(value)
                self.writer.add_scalar(f"train/{key}", mean_value, self.global_step)

            end_info = self.on_epoch_end()
            write_info("train", end_info)

            eval_total.clear()
            for period in val_periods:
                eval_info = self.evaluate(eval_env, period)
                write_info(f"eval/period-{period}", eval_info)

                for key, value in eval_info.items():
                    eval_total[key].append(value)

            for key, value in eval_total.items():
                mean_value = np.nanmean(value)
                self.writer.add_scalar(f"eval/{key}", mean_value, self.global_step)

            if self.checkpoint_interval is not None and epoch % self.checkpoint_interval == 0:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), self.checkpoint_dir / f'checkpoint_{epoch}.pth')
                torch.save(self.actor.state_dict(), self.checkpoint_dir / f'actor_checkpoint_{epoch}.pth')

            if lr_scheduler is not None:
                lr_scheduler.step()


        if hasattr(self, "run"):
            self.run.finish()

        self.writer.close()

        if running_experiment:
            save_dir = get_root_path() / "saved_models" / self.experiment_name
            save_dir.mkdir(parents=True, exist_ok=True)

            torch.save(self.state_dict(), save_dir / 'algorithm_final.pth')
            torch.save(self.actor.state_dict(), save_dir / 'actor_final.pth')

            if hasattr(self, "config") and self.config is not None:
                import yaml

                self.config["saved_models"] = {
                    'full'  : str((save_dir / 'algorithm_final.pth').relative_to(get_root_path())),
                    'actor' : str((save_dir / 'actor_final.pth').relative_to(get_root_path())),
                }

                with open(save_dir / 'config.yaml', 'w') as f:
                    yaml.dump(self.config, f)


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
                bids, action, entropy = env.strategy.get_bid_action(**obs)

                obs, reward, done, info = env.step(bids)

                cum_reward += reward

        info[f"total_reward"] = cum_reward

        env.strategy.reset()

        return info
