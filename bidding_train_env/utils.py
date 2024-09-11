"""
    utils.py

General utility functions that does not depend on any object in the project.
"""
from typing import Any, Sequence, Callable

from pathlib import Path

from omegaconf import OmegaConf

import torch.optim as optim
from torch.nn import Module
import torch

def get_root_path():
    return Path(__file__).parent.parent


def set_seed(seed: int):
    import random, torch
    import numpy as np

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def unstack_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    unstacked_parameters = {}

    for parameter, value in parameters.items():
        if '.' in parameter:
            nesting = parameter.split('.')

            current_dict = unstacked_parameters

            for nest in nesting[:-1]:
                if nest not in current_dict:
                    current_dict[nest] = {}

                current_dict = current_dict[nest]
        
            current_dict[nesting[-1]] = value
        
        else:
            unstacked_parameters[parameter] = value

    return unstacked_parameters


def config_to_dict(config) -> dict:
    container = OmegaConf.to_container(config)

    if isinstance(container, list):
        return {i: value for i, value in enumerate(container)}

    elif isinstance(container, dict):
        return container

    else:
        raise ValueError(f"Unsupported type {type(container)}")


# TODO: Bring validate_config here.

# wandb utils

def fix_sweeps(sweep_config: dict[Any, Any]):
    parameters = sweep_config['parameters']

    unstacked_parameters = unstack_parameters(parameters)

    sweep_config["parameters"] = _recursively_fix(unstacked_parameters)

    return sweep_config


def _recursively_fix(parameters: dict[Any, Any]) -> dict[Any, Any]:
    fixed_parameters = {}

    for parameter, value in parameters.items():
        if not isinstance(value, dict):
            fixed_parameters[parameter] = {"value" : value}

        # Check for leaf parameters
        elif ("value" in value) or ("values" in value) or ("distribution" in value):
            fixed_parameters[parameter] = value

        else:
            fixed_parameters[parameter] = {"parameters": _recursively_fix(value)}

    return fixed_parameters


def get_optimizer(
        model: Module | Sequence[Module],
        optimizer_name: str,
        **kwargs
    ) -> optim.Optimizer:


    optimizer_cls = getattr(optim, optimizer_name)
    
    if isinstance(model, Sequence):
        return optimizer_cls([param for m in model for param in m.parameters()], **kwargs)
    
    else:
        return optimizer_cls(model.parameters(), **kwargs)


def get_activation(activation: Callable[[torch.Tensor], torch.Tensor] | str | None) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation is None:
        return lambda x: x

    if activation == 'softmax':
        return torch.nn.Softmax(dim=-1)

    if isinstance(activation, str):
        activation_fn = getattr(torch.nn.functional, activation, None)

        if activation_fn is not None and isinstance(activation_fn(torch.randn(1,2)), torch.Tensor):
            return activation_fn

        activation_fn = getattr(torch.nn, activation, None)

        if activation_fn is not None and isinstance(activation_fn()(torch.randn(1,2)), torch.Tensor):
            return activation_fn


        raise ValueError(f"Activation function {activation} not found")
    
    else:
        return activation


def turn_off_grad(model: Module):
    model.eval()

    for param in model.parameters():
        param.requires_grad = False


def discounted_returns(rewards: torch.Tensor, gamma=0.99):
    n_samples, samples_len = rewards.shape

    gammas = 0.99 ** torch.arange(samples_len, device=rewards.device)

    discounted_rewards = rewards * gammas

    return (discounted_rewards + discounted_rewards.sum(-1, keepdim=True) - discounted_rewards.cumsum(-1)) / gammas