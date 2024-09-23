from typing import Sequence

import torch
from torch.optim import Optimizer
from torch.nn.functional import mse_loss

from tensordict import TensorDict

from .base_algo import RLAlgorithm
from ..agents import Actor, Critic


from ..utils import turn_off_grad

from copy import deepcopy

class TD3BC(RLAlgorithm):
    def __init__(
            self,
            actor: Actor,
            critic_ensemble: Sequence[Critic],
            actor_optimizer: Optimizer,
            critic_optimizer: Optimizer,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: float = 0.2,
            noise_std: float = 0.2,
            noise_clip: float = 0.5,
            actor_update_freq: int = 2,
        ): 
        super().__init__()

        self.actor                  = actor
        self.target_actor           = deepcopy(actor)

        turn_off_grad(self.target_actor)

        self.critic_ensemble        = critic_ensemble
        self.target_critic_ensemble = deepcopy(critic_ensemble)

        for critic in self.target_critic_ensemble:
            turn_off_grad(critic)

        self.actor_optimizer  = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.noise_std = noise_std
        self.noise_clip = noise_clip

        self.actor_update_freq = actor_update_freq


    def train(self, mode: bool = True):
        self.training = mode

        self.actor.train(mode)

        for critic in self.critic_ensemble:
            critic.train(mode)

        return self


    def train_step(self, batch: TensorDict):
        obs           = batch['state']
        target_action = batch['action']
        rewards       = batch['reward']
        next_obs      = batch['next_state']
        dones         = batch['done']

 
        q_ensemble = torch.stack([
            critic.get_q_value(obs, target_action) for critic in self.critic_ensemble
        ], dim=-1)

        noise = torch.randn_like(target_action) * self.noise_std
        noise = noise.clamp(-self.noise_clip, self.noise_clip)

        next_action, _, _ = self.target_actor.get_action(next_obs)
        next_action = next_action.view(target_action.size())
        next_action += noise

        target_q_ensemble = torch.stack([
            critic.get_q_value(next_obs, next_action) for critic in self.target_critic_ensemble
        ], dim=-1)

        target_q = target_q_ensemble.min(dim=-1).values
        target_q = rewards + self.gamma * (~dones) * target_q
        target_q = target_q.unsqueeze(-1)

        critic_loss = ((q_ensemble - target_q)**2).mean(dim=0).sum()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = torch.tensor(torch.nan)
        if self.global_step % self.actor_update_freq == 0:
            sampled_action, _, _ = self.actor.get_action(obs)
            sampled_action = sampled_action.view(target_action.size())

            q_value = self.critic_ensemble[0].get_q_value(obs, sampled_action)

            lmbda = self.alpha / q_value.detach().abs().mean()

            actor_loss = -lmbda * q_value.mean() + mse_loss(sampled_action, target_action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        

        return {
            'loss/actor': actor_loss.item(),
            'loss/critic': critic_loss.item()
        }

    def save(self):
        return {
            'actor': self.actor.state_dict(),
            'critic_ensemble': [critic.state_dict() for critic in self.critic_ensemble],
            'algorithm': self.state_dict()
        }