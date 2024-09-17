from typing import Sequence, Literal, Optional
from torch.types import Device

Shape = Sequence[int]

from warnings import warn

import torch

from tensordict import TensorDict
from pandas import DataFrame

from .utils import discounted_returns

RB_KEYS  = Literal['state', 'action', 'reward', 'next_state', 'done']
ERB_KEYS = Literal['states', 'actions', 'rewards', 'rtgs', 'timesteps', 'dones', 'mask']


class AbstractReplayBuffer:
    def push(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, batch_size: int) -> TensorDict:
        raise NotImplementedError

    def normalize(self, *args, **kwargs):
        raise NotImplementedError

    def to(self, device: Device):
        raise NotImplementedError

    def __getitem__(self, key) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ReplayBuffer(AbstractReplayBuffer):
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

        self.reward_scale = torch.tensor(1., dtype=torch.float32, device=device)


    @classmethod
    def from_data(
            cls,
            data: DataFrame,
            reward: Optional[str] = None,
            device: Device = 'cpu'
        ):

        capacity = len(data)

        observation_shape = data['state'].iloc[0].shape
        action_shape      = data['action'].iloc[0].shape

        self = cls(capacity, observation_shape, action_shape, device)

        reward_data = data['reward'] if reward is None else data['reward', reward]

        self.push(
            torch.tensor(data['state'].to_numpy(), dtype=torch.float32),
            torch.tensor(data['action'].to_numpy(), dtype=torch.float32),
            torch.tensor(reward_data.to_numpy(), dtype=torch.float32),
            torch.tensor(data['next_state'].to_numpy(), dtype=torch.float32),
            torch.tensor(data['done'].to_numpy(), dtype=torch.bool)
        )

        return self


    def to(self, device: Device):
        self.device = device

        self.buffer = self.buffer.to(device)

        self.state_normalization['mean'] = self.state_normalization['mean'].to(device)
        self.state_normalization['std']  = self.state_normalization['std'].to(device)

        self.reward_scale = self.reward_scale.to(device)

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
        sampled_buffer: TensorDict = self.buffer[idxs]

        state_mean = self.state_normalization['mean']
        state_std  = self.state_normalization['std']

        sampled_buffer["state"]      = (sampled_buffer["state"] - state_mean) / state_std
        sampled_buffer["next_state"] = (sampled_buffer["next_state"] - state_mean) / state_std
        sampled_buffer["reward"]     = sampled_buffer["reward"] / self.reward_scale

        return sampled_buffer


    def normalize(
            self,
            state_mean   : Optional[torch.Tensor] = None,
            state_std    : Optional[torch.Tensor] = None,
            reward_scale : Optional[torch.Tensor] = None,
        ):
        self.state_normalization['mean'] = \
            self['state'].mean(dim=0) if state_mean is None else state_mean

        self.state_normalization['std'] = \
            self['state'].std(dim=0) if state_std is None else state_std

        # Fix me: Hardcoded for this environment
        self.reward_scale = \
            48 * self['reward'].abs().max() if reward_scale is None else reward_scale



class EpisodeReplayBuffer(AbstractReplayBuffer):
    def __init__(
            self,
            capacity: int,
            max_ep_len: int,
            observation_shape: Shape,
            action_shape:      Shape = (),
            window_size: Optional[int] = None,
            gamma: float = 0.99,
            device: Device = 'cpu',
            return_priority: bool = False
        ):
        self.max_ep_len = max_ep_len
        self.window_size = window_size

        self.capacity = capacity
        self.device   = device

        self.observation_shape = observation_shape
        self.action_shape      = action_shape

        self.gamma = gamma

        self.current_size = 0
        self.pointer      = 0

        self.buffer = TensorDict({
            "states"    : torch.zeros((capacity, max_ep_len, *observation_shape), dtype=torch.float32),
            "actions"   : torch.zeros((capacity, max_ep_len, *action_shape),      dtype=torch.float32),
            "rewards"   : torch.zeros((capacity, max_ep_len),                     dtype=torch.float32),
            "rtgs"      : torch.zeros((capacity, max_ep_len),                     dtype=torch.float32),
            "timesteps" : torch.zeros((capacity, max_ep_len),                     dtype=torch.long   ),
            "dones"     : torch.ones((capacity, max_ep_len),                      dtype=torch.bool   ),
            "mask"      : torch.ones((capacity, max_ep_len),                      dtype=torch.bool   ),
        }, batch_size=(capacity, max_ep_len), device=device)


        self.sizes   = torch.zeros((capacity,), dtype=torch.long)
        self.returns = torch.zeros((capacity,), dtype=torch.float32)


        self.state_normalization = {
            'mean': torch.zeros((*observation_shape,), dtype=torch.float32, device=device),
            'std' : torch.ones((*observation_shape,), dtype=torch.float32, device=device)
        }

        self.reward_scale = torch.tensor(1., dtype=torch.float32, device=device)

        self.return_priority = return_priority

        if window_size is not None:
            template = torch.cat([
                torch.arange(window_size).repeat(window_size, 1),
                torch.arange(1, max_ep_len - window_size + 1).unsqueeze(-1) + torch.arange(window_size)
            ])

            mask_template = torch.ones_like(template, dtype=torch.bool)

            self.template      = torch.tril(template, diagonal=0)
            self.mask_template = torch.triu(mask_template, diagonal=1)



    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, current_size={self.current_size})"


    def __len__(self):
        return self.current_size


    def __getitem__(self, key: ERB_KEYS) -> torch.Tensor:
        return self.buffer[key][:self.current_size]

    def to(self, device: Device):
        self.device = device

        self.buffer = self.buffer.to(device)

        self.state_normalization['mean'] = self.state_normalization['mean'].to(device)
        self.state_normalization['std']  = self.state_normalization['std'].to(device)

        self.reward_scale = self.reward_scale.to(device)

        return self


    def push(
            self,
            states : torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            dones  : torch.Tensor,
            trajectory_ids: Optional[torch.Tensor] = None
        ):

        if trajectory_ids is None:
            trajectory_ids = torch.zeros(states.shape[0], dtype=torch.int32)

        states  = states.view(-1, *self.observation_shape)
        actions = actions.view(-1, *self.action_shape)
        rewards = rewards.view(-1)
        dones   = dones.view(-1)

        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == dones.shape[0], \
            "All tensors must have the same lenght"
        
        unique_idxs, idxs, counts = trajectory_ids.unique(return_inverse=True, return_counts=True)

        n_trajectories = len(unique_idxs)

        if n_trajectories > self.capacity:
            warn(f"Trying to push more experiences than the buffer capacity.")

        idxs = (self.pointer + idxs) % self.capacity

        # Bottleneck O(n²)
        columns = (trajectory_ids.unsqueeze(-1) == unique_idxs).cumsum(dim=0) \
            .gather(1, trajectory_ids.unsqueeze(-1)).squeeze() - 1

        trajectory_idxs = (self.pointer + torch.arange(n_trajectories, device=self.device)) % self.capacity

        self.buffer["states"][idxs, columns]    = states.to(self.device)
        self.buffer["actions"][idxs, columns]   = actions.to(self.device)
        self.buffer["rewards"][idxs, columns]   = rewards.to(self.device)
        self.buffer["dones"][idxs, columns]     = dones.to(self.device)
        self.buffer["mask"][idxs, columns]      = False

        self.buffer["timesteps"][idxs, columns] = columns.to(self.device)
        self.buffer["rtgs"][trajectory_idxs]    = discounted_returns(self.buffer["rewards"][trajectory_idxs], self.gamma)

        self.sizes[trajectory_idxs]    = counts
        self.returns[trajectory_idxs]  = self.buffer["rewards"][trajectory_idxs].sum(dim=1).cpu()

        self.current_size = min(self.current_size + n_trajectories, self.capacity)
        self.pointer      = (self.pointer + n_trajectories) % self.capacity



    def sample(self, batch_size: int) -> TensorDict:
        if self.return_priority:
            probs = self.returns / self.returns.sum()

            trajectory_idxs = torch.multinomial(probs, batch_size, replacement=False)
        
        else:
            trajectory_idxs = torch.randint(0, self.current_size, (batch_size,))
        

        sampled_buffer: TensorDict = self.buffer[trajectory_idxs]

        if self.window_size is None:
            return sampled_buffer

        # Sample subtrajectories
        sizes = self.sizes[trajectory_idxs]

        end_times = torch.floor(sizes * torch.rand(batch_size)).long()

        timesteps = self.template[end_times].to(self.device)

        sampled_buffer = sampled_buffer.gather(1, timesteps)

        sampled_buffer["mask"] |= self.mask_template[end_times].to(self.device)

        state_mean = self.state_normalization['mean']
        state_std  = self.state_normalization['std']

        sampled_buffer["states"]      = (sampled_buffer["states"] - state_mean) / state_std
        sampled_buffer["rewards"]     = sampled_buffer["rewards"] / self.reward_scale

        return sampled_buffer


    def normalize(
            self,
            state_mean   : Optional[torch.Tensor] = None,
            state_std    : Optional[torch.Tensor] = None,
            reward_scale : Optional[torch.Tensor] = None,
        ):
        self.state_normalization['mean'] = \
            self['states'].mean(dim=0) if state_mean is None else state_mean

        self.state_normalization['std'] = \
            self['states'].std(dim=0) if state_std is None else state_std

        max_return = (self.returns * self.sizes).max()

        self.reward_scale = \
            max_return if reward_scale is None else reward_scale
