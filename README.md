# Overview
This is an auto-bidding training framework refactored from the [original repository](https://github.com/alimama-tech/NeurIPS_Auto_Bidding_General_Track_Baseline) to be a modular and easier-to-develop framework. This repository includes four modules that work together for training bidding agents, structured as below:
```
bidding_train_env/  
│   import_utils.py
│   replaybuffer.py
│   utils.py
│
├── agents/
│   ├── actor/
│   │       base.py
│   │       mlp.py
│   │       transformer.py
│   │
│   ├── critic/
│   │       base.py
│   │       mlp.py
│   │
│   └── value/
│           base.py
│           mlp.py
│
├── algorithms/
│       base_algo.py
│       behavior_cloning.py
│       decision_transformer.py
│       iql.py
│       td3bc.py
│
├── envs/
│       base.py
│       offline_env.py
│
└── strategy/
        base_bidding_strategy.py
        simple_strategy.py
```

- **agents:** Include common neural networks and parameterized function approximators used on RL algorithms. This module have 3 submodules: _actor_, _critic_ and _value_.

- **strategy:** High-level abstraction for a bidding agent. Responsible for the logic of state representation, on which the actor can infer, and the action decoding, which transforms the output of the actor into the proper bids.

- **envs:** Environments that simulates auction and impressions, following the protocol of [Gymnasium](https://gymnasium.farama.org/) environments.

- **algorithms:** Implementations of algorithms to train bidding policies. The implementations are general enough to accept different strategies and agents with different architectures.


## Dependencies
To develop new strategies and algorithms, install the requirements for the environment, and also install the package as editable, using the following
```
conda create -n auto-bidding python=3.11.9 pip=24.2
conda activate auto-bidding
pip install -r requirements.txt
pip install -e .
```

# Usage
## Download data
The data used for this challenge is tabular data containing information about the advertisers, auction bids and results for each opportunity, disposed on different timesteps and periods. To download, preprocess and use the data, run the scripts on the `/data` directory as following:

- `download.py` downloads the raw data into .csv files, separated by period intervals
- `efficient_repr.py` uses the raw data to separate the information into 3 different representations: Advertisers data, Bidding data and Impression data, which are encoded to be processed quicker and using less hard-drive space.
- `rl_data_generator.py` pre-process the data according to the `SimpleBiddingStrategy`. It is an initial step for training simple bidding agents based on a simple state-action processing.

## Running algorithms
Every algorithm implemented on `algorithms` have a run script to start an experiment, which can be controled by a configuration file, dependent on the algorithm choosen.

To run an algorithm with default configuration, run
```bash
python run/run_{algorithm name}.py
```

If you have [wandb](https://wandb.ai/home) working on your machine, any results will be exported, and the metrics of training can be tracked in your browser. If you want to modify the current configuration for running an algorithm, many .yaml config files can be found on `/run` and can be modified. To run a custom configuration, please run the following command

```bash
python run/run_{algorithm name}.py {path to config}.yaml
```

## Sweeping
Wandb integration does not only allow easier experiment tracking, but hyperparameter tuning, by the name of *sweeps*.

Work in progress.
