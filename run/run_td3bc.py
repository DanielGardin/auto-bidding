from typing import Optional, Any, cast

from pathlib import Path

import pandas as pd

import torch

from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING

from bidding_train_env.utils import get_root_path, set_seed, get_optimizer,config_to_dict
from bidding_train_env.import_utils import get_actor, get_critic, get_strategy, get_env
from bidding_train_env.algorithms import TD3BC
from bidding_train_env.replaybuffer import ReplayBuffer

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
    experiment_name: str = "td3bc"

@dataclass
class DataParams:
    data_dir: str          = 'data/traffic/rl_data/rl_data.parquet'
    buffer_size: int       = MISSING
    val_periods: list[int] = field(default_factory=list)

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
    critic: str                   = MISSING
    critic_params: dict[str, Any] = field(default_factory=dict)
    num_critics: int              = 2

@dataclass
class TrainParams:
    batch_size: int                         = MISSING
    num_epochs: int                         = MISSING
    steps_per_epoch: int                    = MISSING
    actor_optimizer: str                    = MISSING
    actor_optimizer_params: dict[str, Any]  = field(default_factory=lambda : dict(lr=MISSING))
    critic_optimizer: str                   = MISSING
    critic_optimizer_params: dict[str, Any] = field(default_factory=lambda : dict(lr=MISSING))
    tau: float                              = 0.005
    gamma: float                            = 0.99
    alpha: float                            = 0.2
    noise_std: float                        = 0.2
    noise_clip: float                       = 0.5
    actor_update_freq: int                  = 2

@dataclass
class LoggingParams:
    log_dir: str                       = 'logs/td3bc'
    log_interval: Optional[int]        = None
    checkpoint_interval: Optional[int] = None
    use_wandb: bool                    = False
    verbose: bool                      = False

@dataclass
class TD3BCParams:
    general: GeneralParams         = field(default_factory=GeneralParams)
    data: DataParams               = field(default_factory=DataParams)
    environment: EnvironmentParams = field(default_factory=EnvironmentParams)
    model: ModelParams             = field(default_factory=ModelParams)
    train: TrainParams             = field(default_factory=TrainParams)
    logging: LoggingParams         = field(default_factory=LoggingParams)

template = TD3BCParams()

def validate_config(parameters):
    if isinstance(parameters, dict):
        params = OmegaConf.create(parameters)
    
    elif isinstance(parameters, (str, Path)):
        params = OmegaConf.load(parameters)
    
    else:
        raise ValueError("Invalid parameters type")

    validated_config = OmegaConf.merge(template, params)
    validated_config = cast(TD3BCParams, validated_config)

    return validated_config


default_config = {
    "general" : {
        "seed"        : 42,
        "device"      : "cuda",
        "project_name": "bidding_train_env",
    },
    "data" : {
        "data_dir"         : 'data/traffic/rl_data/rl_data.parquet',
        "buffer_size"      : 50_000,
        "val_periods"      : [25, 26, 27],
    },
    "environment" : {
        "environment" : "OfflineBiddingEnv",
        "observation_shape" : (16,),
        "action_shape" : (),
    },
    "model" : {
        "actor"  : "ContinousStochasticMLP",
        "actor_params" : {
            "input_dim" : 16,
            "hidden_dims" : [256, 256],
            "output_dim" : 1,
            "activation" : "relu"
        },
        "strategy" : "SimpleBiddingStrategy",
        "critic" : "QEmbedMLP",
        "critic_params" : {
            "observation_shape" : (16,),
            "action_shape" : (),
            "embedding_dim": 64,
            "hidden_dims" : [256, 256],
            "activation" : "relu"
        },
        "num_critics" : 2,
    },
    "train" : {
        "batch_size" : 100,
        "num_epochs": 10_000,
        "steps_per_epoch" : 100,
        "actor_optimizer" : "Adam",
        "actor_optimizer_params" : {
            "lr"          : 1e-4,
        },
        "critic_optimizer" : "Adam",
        "critic_optimizer_params" : {
            "lr"          : 1e-4,
        },
        "tau"              : 0.005,
        "gamma"            : 0.99,
        "alpha"            : 0.2,
        "noise_std"        : 0.2,
        "noise_clip"       : 0.5,
        "actor_update_freq": 2
    },
    "logging" : {
        "log_dir"            : "logs/td3bc",
        "checkpoint_interval": 1000,
        "use_wandb"          : True,
    }
}

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

    if params.general.seed is not None: set_seed(params.general.seed)

    if params.general.device == 'auto':
        params.general.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    actor = get_actor(params.model.actor, **params.model.actor_params)
    actor = actor.to(params.general.device)

    critic_ensemble = [
        get_critic(params.model.critic, **params.model.critic_params).to(params.general.device) for _ in range(params.model.num_critics)
    ]

    actor_optimizer  = get_optimizer(actor, params.train.actor_optimizer, **params.train.actor_optimizer_params)
    critic_optimizer = get_optimizer(critic_ensemble, params.train.critic_optimizer, **params.train.critic_optimizer_params)

    td3bc = TD3BC(
        actor,
        critic_ensemble,
        actor_optimizer,
        critic_optimizer,
        tau=params.train.tau,
        gamma=params.train.gamma,
        alpha=params.train.alpha,
        noise_std=params.train.noise_std,
        noise_clip=params.train.noise_clip,
        actor_update_freq=params.train.actor_update_freq
    )

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

    replay_buffer = ReplayBuffer(
        capacity          = params.data.buffer_size,
        observation_shape = params.environment.observation_shape,
        action_shape      = params.environment.action_shape,
        device            = params.general.device
    )

    replay_buffer.push(
        torch.tensor(data['state'].to_numpy(), dtype=torch.float32),
        torch.tensor(data['action'].to_numpy(), dtype=torch.float32),
        torch.tensor(data['reward', 'continuous'].to_numpy(), dtype=torch.float32),
        torch.tensor(data['next_state'].to_numpy(), dtype=torch.float32),
        torch.tensor(data['done'].to_numpy(), dtype=torch.bool)
    )

    td3bc.begin_experiment(
        project_name = params.general.project_name,
        experiment_name = params.general.experiment_name,
        log_dir = params.logging.log_dir,
        checkpoint_interval = params.logging.checkpoint_interval,
        use_wandb = params.logging.use_wandb,
        config = config_to_dict(params)
    )

    td3bc.learn(
        num_epochs=params.train.num_epochs,
        steps_per_epoch=params.train.steps_per_epoch,
        replay_buffer=replay_buffer,
        batch_size=params.train.batch_size,
        eval_env=env,
        val_periods=params.data.val_periods,
    )