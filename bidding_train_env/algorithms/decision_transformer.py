# from torch.nn.functional import mse_loss
import torch
from torch.optim import Optimizer

from tensordict import TensorDict

from .base_algo import RLAlgorithm
from ..agents import Actor


class DecisionTransformer(RLAlgorithm):
    def __init__(
            self,
            actor: Actor,
            optimizer: Optimizer,
        ):
        super().__init__()

        self.actor = actor
        self.optimizer = optimizer


    def train_step(self, batch: TensorDict):
        # action, _, _ = self.actor.get_action(batch['state'])
        states          = batch['state']
        target_action   = batch['action']
        rewards         = batch['reward']
        dones           = batch['done']
        rtg             = batch['rtg']
        attention_mask  = batch['attention_mask']

        state_target    = torch.clone(states)
        action_target   = torch.clone(target_action)
        reward_target   = torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.actor.forward(
            states, target_action, rewards, attention_mask=attention_mask, 
            target_return=rtg[:,0],)

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:,-1].reshape(-1, act_dim)

        loss = ((action_preds - target_action)**2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss/action" : loss.item()
        }
