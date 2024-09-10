from torch.nn.functional import mse_loss
from torch.optim import Optimizer

from tensordict import TensorDict

from .base_algo import RLAlgorithm
from ..agents import Actor


class BehaviorCloning(RLAlgorithm):
    def __init__(
            self,
            actor: Actor,
            optimizer: Optimizer,
        ):
        super().__init__()

        self.actor = actor
        self.optimizer = optimizer


    def train_step(self, batch: TensorDict):
        action, _, _ = self.actor.get_action(batch['state'])

        target_action = batch['action']
        action        = action.view(target_action.size())

        loss = mse_loss(action, target_action)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss/action" : loss.item()
        }
