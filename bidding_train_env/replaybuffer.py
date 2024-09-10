from typing import Sequence, Literal, Optional
from torch.types import Device

Shape = Sequence[int]

from warnings import warn

import torch

from tensordict import TensorDict

RB_KEYS = Literal['state', 'action', 'reward', 'next_state', 'done']

class ReplayBuffer:
    def __init__(
            self,
            capacity: int,
            observation_shape: Shape,
            action_shape:      Shape = (),
            device: Device = 'cpu',
        ):
        self.capacity = capacity
        self.device   = device

        self.observation_shape = observation_shape
        self.action_shape      = action_shape

        self.current_size = 0
        self.pointer      = 0

        self.buffer = TensorDict({
            'state'     : torch.empty((capacity, *observation_shape), dtype=torch.float32),
            'action'    : torch.empty((capacity, *action_shape),      dtype=torch.float32),
            'reward'    : torch.empty((capacity,),                    dtype=torch.float32),
            'next_state': torch.empty((capacity, *observation_shape), dtype=torch.float32),
            'done'      : torch.empty((capacity,),                    dtype=torch.bool   ),
        }, batch_size=(capacity,), device=device)

        self.state_normalization = {
            'mean': torch.zeros((*observation_shape,), dtype=torch.float32, device=device),
            'std' : torch.ones((*observation_shape,), dtype=torch.float32, device=device)
        }

        self.reward_normalization = {
            'mean': torch.tensor(0., dtype=torch.float32, device=device),
            'std' : torch.tensor(1., dtype=torch.float32, device=device)
        }
    
    def to(self, device: Device):
        self.device = device

        self.buffer = self.buffer.to(device)

        self.state_normalization['mean'] = self.state_normalization['mean'].to(device)
        self.state_normalization['std']  = self.state_normalization['std'].to(device)

        self.reward_normalization['mean'] = self.reward_normalization['mean'].to(device)
        self.reward_normalization['std']  = self.reward_normalization['std'].to(device)

        return self


    def __getitem__(self, key: RB_KEYS) -> torch.Tensor:
        return self.buffer[key][:self.current_size]


    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, current_size={self.current_size})"


    def __len__(self):
        return self.current_size

    def push(
            self,
            state     : torch.Tensor,
            action    : torch.Tensor,
            reward    : torch.Tensor,
            next_state: torch.Tensor,
            done      : torch.Tensor
        ):
        assert state.shape[0] == action.shape[0]     \
                              == reward.shape[0]     \
                              == next_state.shape[0] \
                              == done.shape[0],      \
            "All tensors must have the same first dimension"
        
        n_experiences = state.shape[0]

        if n_experiences > self.capacity:
            warn(f"Trying to push more experiences than the buffer capacity.")

        idxs = (self.pointer + torch.arange(n_experiences)) % self.capacity

        self.buffer['state'][idxs]      = state.to(self.device)
        self.buffer['action'][idxs]     = action.to(self.device)
        self.buffer['reward'][idxs]     = reward.to(self.device)
        self.buffer['next_state'][idxs] = next_state.to(self.device)
        self.buffer['done'][idxs]       = done.to(self.device)

        self.current_size = min(self.current_size + n_experiences, self.capacity)
        self.pointer      = (self.pointer + n_experiences) % self.capacity


    def sample(self, batch_size: int) -> TensorDict:
        idxs = torch.randint(0, self.current_size, (batch_size,))
        sampled_buffer = self.buffer[idxs] # type: ignore

        state_mean = self.state_normalization['mean']
        state_std  = self.state_normalization['std']

        reward_mean = self.reward_normalization['mean']
        reward_std  = self.reward_normalization['std']

        sampled_buffer["state"]      = (sampled_buffer["state"] - state_mean) / state_std
        sampled_buffer["next_state"] = (sampled_buffer["next_state"] - state_mean) / state_std
        sampled_buffer["reward"]     = (sampled_buffer["reward"] - reward_mean) / reward_std

        return sampled_buffer


    def normalize(
            self,
            state_mean : Optional[torch.Tensor] = None,
            state_std  : Optional[torch.Tensor] = None,
            reward_mean: Optional[torch.Tensor] = None,
            reward_std : Optional[torch.Tensor] = None
        ):
        self.state_normalization['mean'] = \
            self.buffer['state'].mean(dim=0) if state_mean is None else state_mean

        self.state_normalization['std'] = \
            self.buffer['state'].std(dim=0) if state_std is None else state_std

        self.reward_normalization['mean'] = \
            self.buffer['reward'].mean() if reward_mean is None else reward_mean
        
        self.reward_normalization['std'] = \
            self.buffer['reward'].std() if reward_std is None else reward_std