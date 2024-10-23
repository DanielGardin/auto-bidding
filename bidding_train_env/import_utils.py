from typing import Optional

from .agents import actor, critic, value
from . import envs
from . import strategy

def get_actor(actor_name: str, **kwargs) -> actor.Actor:
    return getattr(actor, actor_name)(**kwargs)

def get_critic(critic_name: str, **kwargs) -> critic.Critic:
    return getattr(critic, critic_name)(**kwargs)

def get_value(value_name: str, **kwargs) -> value.Value:
    return getattr(value, value_name)(**kwargs)


def get_strategy(
        strategy_name: str,
        actor: Optional[actor.Actor] = None,
        budget: float = 100.,
        cpa: float = 2.,
        category: int = 0,
        state_norm = None
    ) -> strategy.BaseBiddingStrategy:
    strategy_cls = getattr(strategy, strategy_name)

    if actor is None:
        return strategy_cls(
            budget=budget,
            cpa=cpa,
            category=category
        )
    
    return strategy_cls(
        actor=actor,
        budget=budget,
        cpa=cpa,
        category=category,
        state_norm=state_norm
    )

def get_env(
        env_name: str,
        strategy: strategy.BaseBiddingStrategy,
        period: int,
        **kwargs
    ) -> envs.BiddingEnv:
    env_cls =  getattr(envs, env_name)

    return env_cls(
        strategy=strategy,
        period=period,
        **kwargs
    )