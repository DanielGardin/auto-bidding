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
        states          = batch['states']
        target_action   = batch['actions']
        rewards         = batch['rewards']
        dones           = batch['dones']
        rtg             = batch['rtgs']
        timesteps       = batch['timesteps']
        attention_mask  = batch['mask']

        _, action_preds, _ = self.actor(
            states, target_action, rtg, timesteps, attention_mask
        )

        action_preds = action_preds.view(target_action.size())

        loss = (action_preds - target_action)**2 * ~attention_mask

        loss = loss.sum() / torch.count_nonzero(~attention_mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss/action" : loss.item()
        }
