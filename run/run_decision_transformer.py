from typing import Optional, Any, cast

from pathlib import Path

import pandas as pd

import torch

from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING

from bidding_train_env.utils import get_root_path, set_seed, get_optimizer, config_to_dict, get_scheduler
from bidding_train_env.import_utils import get_actor, get_strategy, get_env
from bidding_train_env.algorithms import DecisionTransformer
from bidding_train_env.replaybuffer import EpisodeReplayBuffer

import logging

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)

@dataclass
class GeneralParams:
    seed: Optional[int]  = None
    device: str          = "auto"
    project_name: str    = "bidding_train_env"
    project_path: str    = str(get_root_path())
    experiment_name: str = "dt"
    algorithm: str       = "dt"

@dataclass
class DataParams:
    data_dir: str            = 'data/traffic/new_gen_data/.parquet'
    buffer_size: int         = MISSING
    train_periods: list[int] = field(default_factory=list)
    val_periods: list[int]   = field(default_factory=list)

@dataclass
class EnvironmentParams:
    environment: str                   = MISSING
    observation_shape: tuple           = MISSING
    action_shape: tuple                = MISSING
    environment_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelParams:
    actor: str                    = MISSING
    actor_params: dict[str, Any]  = field(default_factory=dict)
    strategy: str                 = MISSING
    budget: Optional[float]       = None
    cpa: Optional[float]          = None
    category: Optional[int]       = None

@dataclass
class TrainParams:
    batch_size: int                         = MISSING
    num_epochs: int                         = MISSING
    steps_per_epoch: int                    = MISSING
    actor_optimizer: str                    = MISSING
    actor_optimizer_params: dict[str, Any]  = field(default_factory=lambda : dict(lr=MISSING))
    scheduler: Optional[str]                = None
    scheduler_params: dict[str, Any]        = field(default_factory=dict)
    trajectory_window: int                  = 20

@dataclass
class LoggingParams:
    log_dir: str                       = 'logs/dt'
    log_interval: Optional[int]        = None
    checkpoint_interval: Optional[int] = None
    use_wandb: bool                    = False
    verbose: bool                      = False

@dataclass
class DTParams:
    general: GeneralParams         = field(default_factory=GeneralParams)
    data: DataParams               = field(default_factory=DataParams)
    environment: EnvironmentParams = field(default_factory=EnvironmentParams)
    model: ModelParams             = field(default_factory=ModelParams)
    train: TrainParams             = field(default_factory=TrainParams)
    logging: LoggingParams         = field(default_factory=LoggingParams)

template = DTParams()


def validate_config(parameters) -> DTParams:
    if isinstance(parameters, dict):
        params = OmegaConf.create(parameters)
    
    elif isinstance(parameters, (str, Path)):
        params = OmegaConf.load(parameters)
    
    else:
        raise ValueError("Invalid parameters type")

    validated_config = OmegaConf.merge(template, params)
    validated_config = cast(DTParams, validated_config)

    return validated_config


default_config = {
    "general" : {
        "seed"        : 42,
        "device"      : "cuda:2",
        "project_name": "sweeps",
    },
    "data" : {
        "data_dir"         : 'data/traffic/rl_data/rl_data.parquet',
        "buffer_size"      : 50_000,
        "train_periods"    : [7, 8, 9, 10],
        "val_periods"      : [11, 12, 13],
    },
    "environment" : {
        "environment" : "OfflineBiddingEnv",
        "observation_shape" : (16,),
        "action_shape" : (),
    },
    "model" : {
        "actor"  : "Transformer",
        "actor_params" : {
            "state_dim" : 16,
            "act_dim" : 1,
            "K" : 20,
            "max_ep_len" : 48,
            "hidden_size" : 64,
            "transformer_num_layers" : 3,
            "nhead" : 1,
            "dim_feedforward" : 256,
            "activation" : "relu",
            "dropout" : 0.1,
        },
        "strategy" : "AlphaBiddingStrategy",
    },
    "train" : {
        "batch_size" : 100,
        "num_epochs": 10_000,
        "steps_per_epoch" : 100,
        "actor_optimizer" : "AdamW",
        "actor_optimizer_params" : {
            "lr"          : 1e-4,
        },
        "scheduler" : "StepLR",
        "scheduler_params" : {
            "step_size" : 1000,
            "gamma" : 0.5
        },
        "trajectory_window" : 20
    },
    "logging" : {
        "log_dir"            : "logs/dt",
        "checkpoint_interval": 1000,
        "use_wandb"          : True,
    }
}


def run(params: DTParams):

    if params.general.seed is not None: set_seed(params.general.seed)

    if params.general.device == 'auto':
        params.general.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    actor = get_actor(params.model.actor, **params.model.actor_params)
    actor = actor.to(params.general.device)

    actor_optimizer  = get_optimizer(actor, params.train.actor_optimizer, **params.train.actor_optimizer_params)

    scheduler = None
    if params.train.scheduler is not None:
        scheduler = get_scheduler(actor_optimizer, params.train.scheduler, **params.train.scheduler_params)


    dt = DecisionTransformer(actor, actor_optimizer)


    advertiser_number = params.environment.environment_params.get('advertiser_number', 0)
    metadata: dict[str, int] = pd.read_csv(
        get_root_path() / 'data/traffic/efficient_repr/advertiser_data.csv',
        index_col='advertiserNumber'
    ).to_dict(orient='index')[advertiser_number]

    if params.model.budget is None:
        params.model.budget = metadata['budget']

    if params.model.cpa is None:
        params.model.cpa = metadata['CPAConstraint']

    if params.model.category is None:
        params.model.category = metadata['advertiserCategoryIndex']

    strategy = get_strategy(
        params.model.strategy,
        actor,
        params.model.budget,
        params.model.cpa,
        params.model.category
    )

    env = get_env(
        params.environment.environment,
        strategy,
        period=7,
        **params.environment.environment_params
    )

    data_dir = Path(params.general.project_path) / params.data.data_dir
    data = pd.read_parquet(data_dir).drop(params.data.val_periods, level="deliveryPeriodIndex")
    data = data.loc[params.data.train_periods]
    data = data.fillna(0)

    replay_buffer = EpisodeReplayBuffer(
        capacity          = params.data.buffer_size,
        max_ep_len        = 48,
        observation_shape = params.environment.observation_shape,
        action_shape      = params.environment.action_shape,
        window_size       = params.model.actor_params["K"],
        gamma             = 1.,
        device            = params.general.device,
        return_priority   = True
    )

    trajectory_idxs = torch.tensor(
        data.groupby(['deliveryPeriodIndex', 'advertiserNumber']).ngroup().to_numpy()
    )

    replay_buffer.push(
        torch.tensor(data['state'].to_numpy(), dtype=torch.float32),
        torch.tensor(data['action'].to_numpy(), dtype=torch.float32),
        torch.tensor(data['reward', 'continuous'].to_numpy(), dtype=torch.float32),
        torch.tensor(data['done'].to_numpy(), dtype=torch.bool),
        trajectory_idxs
    )
    replay_buffer.normalize(
        state_mean=torch.as_tensor(data['state'].mean(), dtype=torch.float32),
        state_std=torch.as_tensor(data['state'].std(), dtype=torch.float32)
    )

    dt.begin_experiment(
        project_name        = params.general.project_name,
        experiment_name     = params.general.experiment_name,
        log_dir             = params.logging.log_dir,
        checkpoint_interval = params.logging.checkpoint_interval,
        use_wandb           = params.logging.use_wandb,
        config              = config_to_dict(params)
    )

    dt.learn(
        num_epochs      = params.train.num_epochs,
        steps_per_epoch = params.train.steps_per_epoch,
        replay_buffer   = replay_buffer,
        batch_size      = params.train.batch_size,
        env             = env,
        lr_scheduler    = scheduler,
        val_periods     = params.data.val_periods,
    )




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?',type=str, help='Path to the config file')
    parser.add_argument('--sweeps', '-s', action='store_true', help='Run sweeps')
    args = parser.parse_args()

    if args.config_path is None:
        logger.info("No config file provided, using default config")

        config = default_config
    
    else:
        config = args.config_path
    
    params = validate_config(config)

    run(params)