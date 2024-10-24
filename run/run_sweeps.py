from typing import Callable

import wandb

from omegaconf import OmegaConf

from bidding_train_env.utils import fix_sweeps, config_to_dict
import run_bc, run_iql, run_td3bc, run_decision_transformer


algos = {
    'bc'   : run_bc,
    'iql'  : run_iql,
    'td3bc': run_td3bc,
    'dt'   : run_decision_transformer
}

def typedict(d: dict):
    return {k: typedict(v) if isinstance(v, dict) else type(v)  for k, v in d.items()}


def set_experiment(
        algo: str,
        name: str,
        log_dir: str
    ) -> Callable[[], None]:

    counter = 0

    def run_experiment():
        nonlocal counter

        run = wandb.init(
            name             = f"{name}_{counter}",
            dir              = log_dir,
            sync_tensorboard = True
        )

        config = dict(run.config)

        params = algos[algo].validate_config(config)
        params.logging.use_wandb = False

        algos[algo].run(params)

        counter += 1

    return run_experiment



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to the sweep config file')

    args = parser.parse_args()

    path = args.config_path

    params = config_to_dict(OmegaConf.load(path))

    project_name    = params['parameters']['general']['project_name']
    experiment_name = params['parameters']['general']['experiment_name']
    algorithm       = params['parameters']['general']['algorithm']
    log_dir         = params['parameters']['logging']['log_dir']

    sweep_config = fix_sweeps(params)
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    run_fn = set_experiment(
        algo = algorithm,
        name = experiment_name,
        log_dir = log_dir
    )

    wandb.agent(sweep_id, function=run_fn)