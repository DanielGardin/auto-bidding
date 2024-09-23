from typing import Optional, Any, cast

from pathlib import Path

import pandas as pd

import torch

from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING

from bidding_train_env.utils import get_root_path, set_seed, get_optimizer, config_to_dict
from bidding_train_env.import_utils import get_actor, get_strategy, get_env
from bidding_train_env.algorithms import DecisionTransformer
from bidding_train_env.replaybuffer import EpisodeReplayBuffer
import wandb

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

@dataclass
class DataParams:
    data_dir: str            = 'data/traffic/rl_data/rl_data.parquet'
    buffer_size: int         = MISSING
    train_periods: list[int] = field(default_factory=list)
    val_periods: list[int]   = field(default_factory=list)

@dataclass
class EnvironmentParams:
    environment: str                   = MISSING
    observation_shape: tuple           = MISSING
    action_shape: tuple                = MISSING
    environment_wandb.config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelParams:
    actor: str                    = MISSING
    actor_wandb.config: dict[str, Any]  = field(default_factory=dict)
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
    actor_optimizer_wandb.config: dict[str, Any]  = field(default_factory=lambda : dict(lr=MISSING))
    trajectory_window: int                  = 20

@dataclass
class LoggingParams:
    log_dir: str                       = 'logs/dt'
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


sweep_config = {
    'method': 'random',  # Escolha o método de busca: 'grid', 'random', ou 'bayes'
    'metric': {
        'name': 'loss',  # Metrica que será otimizada (substitua por sua métrica)
        'goal': 'minimize'  # Pode ser 'minimize' ou 'maximize'
    },
    'parameters': {
        'general.seed': {
            'value': 42  # Valor fixo
        },
        'general.device': {
            'value': 'cuda:2'  # Valor fixo
        },
        'general.project_name': {
            'value': 'bidding_train_env'  # Valor fixo
        },
        'data.data_dir': {
            'value': 'data/traffic/rl_data/rl_data.parquet'  # Valor fixo
        },
        'data.buffer_size': {
            'value': 50_000  # Valor fixo
        },
        'data.train_periods': {
            'value': [7, 8, 9, 10]  # Valor fixo
        },
        'data.val_periods': {
            'value': [11, 12, 13]  # Valor fixo
        },
        'environment.environment': {
            'value': 'OfflineBiddingEnv'  # Valor fixo
        },
        'model.actor': {
            'value': 'Transformer'  # Valor fixo
        },
        'model.actor_wandb.config.state_dim': {
            'value': 16  # Valor fixo
        },
        'model.actor_wandb.config.act_dim': {
            'value': 1  # Valor fixo
        },
        'model.actor_wandb.config.K': {
            'value': 20  # Valor fixo
        },
        'model.actor_wandb.config.max_ep_len': {
            'value': 48  # Valor fixo
        },
        'model.actor_wandb.config.hidden_size': {
            'value': [64, 128, 256]  # Valor fixo
        },
        'model.actor_wandb.config.d_model': {
            'value': 64  # Valor fixo
        },
        'model.actor_wandb.config.transformer_num_layers': {
            'value': 3  # Valor fixo
        },
        'model.actor_wandb.config.nhead': {
            'value': 1  # Valor fixo
        },
        'model.actor_wandb.config.dim_feedforward': {
            'value': 256  # Valor fixo
        },
        'model.actor_wandb.config.activation': {
            'value': 'relu'  # Valor fixo
        },
        'model.actor_wandb.config.dropout': {
            'values': [0.1, 0.2, 0.3]  # Testar diferentes valores de dropout
        },
        'train.batch_size': {
            'values': [64, 100, 128]  # Variar o tamanho do batch
        },
        'train.num_epochs': {
            'values': [1_000, 2_000, 5_000]  # Variar o número de épocas
        },
        'train.steps_per_epoch': {
            'value': 100  # Valor fixo
        },
        'train.actor_optimizer': {
            'value': 'AdamW'  # Valor fixo
        },
        'train.actor_optimizer_wandb.config.lr': {
            'values': [1e-4, 1e-5, 1e-3]  # Testar diferentes taxas de aprendizado
        },
        'train.trajectory_window': {
            'value': 20  # Valor fixo
        },
        'logging.log_dir': {
            'value': 'logs/dt'  # Valor fixo
        },
        'logging.checkpoint_interval': {
            'value': 1000  # Valor fixo
        },
        'logging.use_wandb': {
            'value': True  # Valor fixo
        }
    }
}


def run():
    start = wandb.init()

    if wandb.config.general.seed is not None: set_seed(wandb.config.general.seed)

    if wandb.config.general.device == 'auto':
        wandb.config.general.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    actor = get_actor(wandb.config.model.actor, **wandb.config.model.actor_wandb.config)
    actor = actor.to(wandb.config.general.device)

    actor_optimizer  = get_optimizer(actor, wandb.config.train.actor_optimizer, **wandb.config.train.actor_optimizer_wandb.config)

    dt = DecisionTransformer(actor, actor_optimizer)


    advertiser_number = wandb.config.environment.environment_wandb.config.get('advertiser_number', 0)
    metadata: dict[str, int] = pd.read_csv(
        get_root_path() / 'data/traffic/efficient_repr/advertiser_data.csv',
        index_col='advertiserNumber'
    ).to_dict(orient='index')[advertiser_number]

    if wandb.config.model.budget is None:
        wandb.config.model.budget = metadata['budget']

    if wandb.config.model.cpa is None:
        wandb.config.model.cpa = metadata['CPAConstraint']

    if wandb.config.model.category is None:
        wandb.config.model.category = metadata['advertiserCategoryIndex']

    strategy = get_strategy(
        wandb.config.model.strategy,
        actor,
        wandb.config.model.budget,
        wandb.config.model.cpa,
        wandb.config.model.category
    )

    env = get_env(
        wandb.config.environment.environment,
        strategy,
        period=7,
        **wandb.config.environment.environment_wandb.config
    )

    data_dir = Path(wandb.config.general.project_path) / wandb.config.data.data_dir
    data = pd.read_parquet(data_dir).drop(wandb.config.data.val_periods, level="deliveryPeriodIndex")
    data = data.loc[wandb.config.data.train_periods]

    replay_buffer = EpisodeReplayBuffer(
        capacity          = wandb.config.data.buffer_size,
        max_ep_len        = 48,
        observation_shape = wandb.config.environment.observation_shape,
        action_shape      = wandb.config.environment.action_shape,
        window_size       = wandb.config.train.trajectory_window,
        gamma             = 1.,
        device            = wandb.config.general.device,
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

    dt.learn(
        num_epochs=wandb.config.train.num_epochs,
        steps_per_epoch=wandb.config.train.steps_per_epoch,
        replay_buffer=replay_buffer,
        batch_size=wandb.config.train.batch_size,
        eval_env=env,
        val_periods=wandb.config.data.val_periods,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?',type=str, help='Path to the config file')
    parser.add_argument('--sweeps', '-s', action='store_true', help='Run sweeps')
    args = parser.parse_args()



    sweep_id = wandb.sweep(sweep=sweep_config, project="")
    wandb.agent(sweep_id, function=run, count=10)


