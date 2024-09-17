from torch import Tensor
from torch.optim import Optimizer

from tensordict import TensorDict

from .base_algo import RLAlgorithm
from ..agents import Actor


class BehaviorCloning(RLAlgorithm):
    def __init__(
            self,
            actor: Actor,
            actor_optimizer: Optimizer,
        ):
        super().__init__()

        self.actor = actor
        self.actor_optimizer = actor_optimizer

        if self.actor.is_stochastic():
            def loss_fn(action: Tensor, log_prob: Tensor, target_action: Tensor):
                return - log_prob.mean()

        else:
            def loss_fn(action: Tensor, log_prob: Tensor, target_action: Tensor):
                return (action - target_action).pow(2).mean()

        self.loss_fn = loss_fn


    def train_step(self, batch: TensorDict):
        action, log_prob, _ = self.actor.get_action(batch['state'])

        target_action = batch['action']
        action        = action.view(target_action.size())

        loss = self.loss_fn(action, log_prob, target_action)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "loss/actor" : loss.item()
        }


    def save(self):
        return {
            'actor': self.actor.state_dict(),
            'algorithm': self.state_dict()
        }