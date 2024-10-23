"""
    offline_env.py

This module contains an offline environment for simulating the advertising bidding process, using
collected data. The original evaluator is refactored to behave as a gymnasium environment, allowing
for easy integration with reinforcement learning algorithms.
"""
from typing import Optional
from numpy.typing import NDArray

from pathlib import Path

import pandas as pd
import numpy as np

from .base import BiddingEnv
from ..strategy import BaseBiddingStrategy

from ..utils import get_root_path



def get_score_nips(reward: float, cpa: float, cpa_constraint: float, *, beta: float = 2.) -> float:
    """
    Evaluates the score of a given period based on the reward, CPA, and CPA constraint.    
    """

    if cpa > cpa_constraint:
        return (cpa_constraint / (cpa + 1e-10))**beta * reward

    return reward

class OfflineBiddingEnv(BiddingEnv):
    MIN_BUDGET = 0.1
    def __init__(
            self,
            strategy: BaseBiddingStrategy,
            data : str = "new",
            period: int = 7,
            advertiser_number: int = 0,
        ):
        if data == "new":
            self.DATA_PATH = get_root_path() / 'data/traffic/new_efficient_repr'
        elif data == "old":
            self.DATA_PATH = get_root_path() / 'data/traffic/efficient_repr'
        
        self.impressions = pd.read_parquet(self.DATA_PATH / 'impression_data.parquet')

        self.advertiser_number = advertiser_number
        self.strategy = strategy

        super().__init__(strategy, period)

    def __repr__(self):
        return f"OfflineBiddingEnv"

    @property
    def budget(self):
        return self.strategy.budget

    @property
    def cpa_constraint(self):
        return self.strategy.cpa

    @property
    def category(self):
        return self.strategy.category


    def set_period(self, period: int):
        super().set_period(period)
        if period == self.current_period and hasattr(self, 'period_data'):
            return
        self.period_data = pd.read_parquet(self.DATA_PATH / f'bidding-period-{period}.parquet')
        self.period_data = self.period_data.fillna(0)
        cols = [(i, "bid") for i in range(48)]
        self.all_bids = [self.period_data.loc[i, cols].values for i in range(48)] # small optimization


    def get_obs(self):
        if not self.is_terminal():
            pValues       = self.period_data.loc[self.current_timestep][self.advertiser_number, 'pValue'].to_numpy()
            pValuesSigmas = self.period_data.loc[self.current_timestep][self.advertiser_number, 'pValueSigma'].to_numpy()

        else:
            pValues       = np.array([])
            pValuesSigmas = np.array([])

        obs = {
            "timeStepIndex": self.current_timestep,
            "pValues"      : pValues,
            "pValueSigmas" : pValuesSigmas,
        }

        return obs | self.history


    def get_info(self):
        if self.num_conversions == 0:
            return {
                "conversions" : 0,
                "cpa" : np.inf,
                "wins" : self.num_wins,
                "score" : 0,
            }

        cpa = (self.budget - self.remaining_budget) / self.num_conversions

        return {
            "conversions" : self.num_conversions,
            "cpa" : cpa,
            "wins" : self.num_wins,
            "score" : get_score_nips(self.num_conversions, cpa, self.cpa_constraint),
        }


    def reset(self):
        self.current_timestep = 0

        self.strategy.reset()
        self.remaining_budget = self.budget

        self.max_timesteps: int = self.period_data.index.get_level_values('timeStepIndex').max()

        for key in self.history:
            self.history[key].clear()

        self.num_conversions = 0
        self.num_wins        = 0

        return self.get_obs(), self.get_info()


    def is_terminal(self) -> bool:
        """
            Check if the environment has reached a terminal state.
        """
        return self.current_timestep >= self.max_timesteps or self.remaining_budget < self.MIN_BUDGET


    def get_offline_bids(self):
        bids = self.all_bids[self.current_timestep][:, self.advertiser_number]
        return bids
        
    def get_competitors_bids(self):
        bids = self.all_bids[self.current_timestep][:, np.arange(48) != self.advertiser_number]
        return bids

    def step(self, bids: Optional[NDArray] = None):
        #if self.is_terminal():
        #    return self.get_obs(), 0, self.is_terminal(), self.get_info()

        if bids is None:
            bids = self.get_offline_bids()

        elif len(bids) != len(self.period_data.loc[self.current_timestep]):
            raise ValueError(f"Number of bids ({len(bids)}) does not match number of opportunities ({len(self.period_data.loc[self.current_timestep])})")

        pValues           = self.period_data.loc[self.current_timestep][self.advertiser_number, 'pValue'].to_numpy()
        pValuesSigmas     = self.period_data.loc[self.current_timestep][self.advertiser_number, 'pValueSigma'].to_numpy()
        leastWinningCosts = self.impressions.loc[self.current_period, self.current_timestep][(3, 'cost')].to_numpy() # type: ignore

        competitors_bids = self.get_competitors_bids()
        winning_bids, costs, conversions = self.simulate_ad_bidding(bids, pValues, pValuesSigmas, competitors_bids)


        if costs.sum() > self.remaining_budget - self.MIN_BUDGET:
            bid_mask = self.resolve_overcosted_bids(bids, costs)

            winning_bids = winning_bids * bid_mask
            costs        = costs        * bid_mask
            conversions  = conversions  * bid_mask

        cost = costs.sum()

        self.remaining_budget -= cost
        self.strategy.pay(cost)


        # Update history
        self.history['historyPValueInfo'].append(np.stack([pValues, pValuesSigmas]).T)
        self.history['historyBid'].append(bids)
        self.history['historyAuctionResult'].append(np.stack([winning_bids, winning_bids, costs]).T)
        self.history['historyImpressionResult'].append(np.stack([conversions, conversions]).T)
        self.history['historyLeastWinningCost'].append(leastWinningCosts)

        # Running statistics
        self.num_conversions += conversions.sum()
        self.num_wins += winning_bids.sum()

        self.current_timestep += 1

        obs    = self.get_obs()
        reward = self.strategy.get_reward(**obs)
        done   = self.is_terminal()
        info   = self.get_info()

        return obs, reward, done, info


    # Alternative (deterministic) implementation
    # def resolve_overcosted_bids(self, bids: NDArray, costs: NDArray):
    #     return costs.cumsum() <= self.remaining_budget


    def resolve_overcosted_bids(self, bids: NDArray, costs: NDArray):
        mask = np.ones_like(bids, dtype=bool)

        while bids[mask].sum() > self.remaining_budget:
            drop_ratio = 1 - self.remaining_budget / bids[mask].sum()

            indices = np.where(mask)[0]
            num_drops = max(int(len(indices) * drop_ratio), 1)

            drop_indices = np.random.choice(indices, num_drops, replace=False)
            
            mask[drop_indices] = False

        return mask


    def simulate_ad_bidding(
            self,
            bids: NDArray,
            pValues: NDArray,
            pValueSigmas: NDArray,
            competitors_bids: NDArray,
        ):
        # get which bids I won (in any position)
        lose = bids[:, None] < competitors_bids
        # I won if I lose at most 2 bids
        winning_bids = np.sum(lose, axis=1) <= 2

        # get the highest bid I won
        win_cost = competitors_bids * (1 - lose)
        
        costs = np.max(win_cost, axis=1) * winning_bids

        # suppose that all bids are exposed
        sim_pValues = np.random.normal(loc=pValues, scale=pValueSigmas) * winning_bids
        sim_pValues = np.clip(sim_pValues, 0, 1)

        conversions = np.random.binomial(1, sim_pValues)

        return winning_bids, costs, conversions