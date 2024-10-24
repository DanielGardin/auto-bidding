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
Wandb integration does not only allow easier experiment tracking, but hyperparameter tuning, by the name of *sweeps*. To run an experiment with sweeps create a config file with the structure expected for your algorithm. To turn it into a sweep file, follow the following template:

Indent you parameters with a `parameters` key, the parameter values may change or be sampled from a distribution, but remain the structure unaltered.

You can choose whether a parameter will be fixed or choosen from a list of possible values, or drawn from a specific distribution. If the value is constant, leave it as it is.

If you want to a certain hyperparameter to be part of the search, indicate how they can change


| desired behavior | parameter setting |
| :--------------: | :---------------- |
| list of possible parameters | <pre lang=yaml>optimizer:<br>  values: ["adam", "sgd", "rmsprop"]</pre>
| categorical distribution | <pre lang=yaml>d_model:<br>  values: [64, 256, 1024]<br>  probabilities: [0.6, 0.3, 0.1]</pre>|
| uniform distribution | <pre lang=yaml>num_layers:<br>  min: 1<br>  max: 4</pre> |
| normal distribution | <pre lang=yaml>temperature:<br>  distribution: normal<br>  mu: 3.0<br>  sigma: 1.0</pre> |
| log uniform distribution | <pre lang=yaml>lr:<br>  distribution: log_uniform<br>  min: -2<br>  sigma: 2</pre> |

And many other distributions can be chosen: [See more](https://docs.wandb.ai/guides/sweeps/sweep-config-keys#parameters)

Finally, add the following header, above the `parameters` field:

```yaml
program: run_sweeps.py

name: <your sweep name>

method: <sweep method | grid, random or bayes>

metric:
  goal: <maximize or minimize>
  name: <name of the metric to be optimized>
  target: <goal value | not required>

run_cap: <maximum number of experiments | not required>
```

Sweeps also have support for [Early stopping](https://docs.wandb.ai/guides/sweeps/sweep-config-keys#early_terminate), if you like!

After your sweep config is done, make sure you added the `algorithm` field in the `general` parameters, this will allow for the sweeps to run the algorithm properly. If everything is done, then run

```bash
python sweeps/run_sweeps.py {path to sweep config}.yaml
```
