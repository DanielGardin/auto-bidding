from bidding_train_env.utils import fix_sweeps, config_to_dict

import wandb

from omegaconf import OmegaConf

import run_bc, run_iql, run_td3bc, run_decision_transformer


algos = {
    'bc'   : run_bc,
    'iql'  : run_iql,
    'td3bc': run_td3bc,
    'dt'   : run_decision_transformer
}

def typedict(d: dict):
    return {k: typedict(v) if isinstance(v, dict) else type(v)  for k, v in d.items()}


def set_experiment(algo: str):

    def run_experiment():
        run = wandb.init()

        config = dict(run.config)

        params = algos[algo].validate_config(config)

        params.logging.use_wandb = False

        algos[algo].run(params)

    return run_experiment



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to the sweep config file')

    args = parser.parse_args()

    path = args.config_path

    params = config_to_dict(OmegaConf.load(path))

    project_name = params['parameters']['general']['project_name']
    algorithm    = params['parameters']['general']['algorithm']

    sweep_config = fix_sweeps(params)
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    wandb.agent(sweep_id, function=set_experiment(algorithm))