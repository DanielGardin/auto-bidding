import torch
import torch.nn as nn
from torch.optim import Optimizer

from tensordict import TensorDict

from .base_algo import RLAlgorithm
from ..agents import Actor, Critic, Value

from copy import deepcopy


class IQL(RLAlgorithm):
    def __init__(
            self,
            actor: Actor,
            critic_ensemble: list[Critic],
            value_net: Value,
            actor_optimizer: Optimizer,
            critic_optimizer: Optimizer,
            value_optimizer: Optimizer,
            tau: float = 0.005,
            gamma: float = 0.99,
            expectile: float = 0.5,
            temperature: float = 0.1,
        ):
        super().__init__()

        self.actor = actor
        self.critic_ensemble     = critic_ensemble
        self.target_critic_ensemble = deepcopy(critic_ensemble)

        self.value_net = value_net

        self.actor_optimizer  = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.value_optimizer  = value_optimizer

        self.tau = tau
        self.gamma = gamma
        self.expectile = expectile
        self.temperature = temperature
    

    def train(self, mode: bool = True):
        self.training = mode

        self.actor.train(mode)

        for critic in self.critic_ensemble:
            critic.train(mode)

        self.value_net.train(mode)
    
        return self



    def train_step(self, batch: TensorDict):
        obs           = batch['state']
        target_action = batch['action']
        rewards       = batch['reward']
        next_obs      = batch['next_state']
        dones         = batch['done']

        with torch.no_grad():
            target_q_ensemble = torch.stack([
                critic.get_q_value(obs, target_action) for critic in self.target_critic_ensemble
            ], dim=-1)

            target_q_values = target_q_ensemble.min(dim=-1).values

        # Value update
        value = self.value_net.get_value(obs)

        advantages = target_q_values - value

        weight = torch.where(advantages > 0, self.expectile, 1 - self.expectile)

        value_loss = (weight * advantages**2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Critic update
        q_ensemble = torch.stack([
            critic.get_q_value(obs, target_action) for critic in self.critic_ensemble
        ], dim=-1)


        with torch.no_grad():
            target_values = self.value_net.get_value(next_obs)

            target_values = rewards + self.gamma * (~dones) * target_values
            target_values = target_values.unsqueeze(-1)


        critic_loss = ((q_ensemble - target_values)**2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # Actor update
        advantages = advantages.detach()

        exp_advantages = torch.exp(advantages * self.temperature)
        exp_advantages = torch.clip(exp_advantages, max=100.)

        _, log_prob, _ = self.actor.get_action(obs)

        actor_loss = (exp_advantages * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # Target update
        for target, critic in zip(self.target_critic_ensemble, self.critic_ensemble):
            for target_param, param in zip(target.parameters(), critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return {
            "loss/actor" : actor_loss.item(),
            "loss/value" : value_loss.item(),
            "loss/critic" : critic_loss.item()
        }