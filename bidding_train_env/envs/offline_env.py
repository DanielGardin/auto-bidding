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
from ..utils import get_root_path



def get_score_nips(reward: float, cpa: float, cpa_constraint: float, *, beta: float = 2.) -> float:
    """
    Evaluates the score of a given period based on the reward, CPA, and CPA constraint.    
    """

    if cpa > cpa_constraint:
        return (cpa_constraint / (cpa + 1e-10))**beta * reward

    return reward

class OfflineBiddingEnv:
    MIN_BUDGET = 0.1
    DATA_PATH = get_root_path() / 'data/traffic/efficient_repr'

    def __init__(
            self,
            advertiser_number  : int,
            advertiser_category: Optional[int]   = None,
            budget             : Optional[float] = None,
            cpa_constraint     : Optional[float] = None,
        ):
        self.impressions = pd.read_parquet(self.DATA_PATH / 'impression_data.parquet')

        # Agent metadata
        self.advertiser_number  = advertiser_number

        metadata = pd.read_csv(
            self.DATA_PATH / 'advertiser_data.csv',
            index_col='advertiserNumber'
        ).to_dict(orient='index')[self.advertiser_number]

        if advertiser_category is None:
            advertiser_category = int(metadata["advertiserCategoryIndex"])
        
        if budget is None:
            budget = float(metadata['budget'])
        
        if cpa_constraint is None:
            cpa_constraint = float(metadata['CPAConstraint'])

        self.category       = advertiser_category
        self.budget         = budget
        self.cpa_constraint = cpa_constraint

        # Agent internal state
        self.remaining_budget = budget
        self.history = {
            'historyPValueInfo'       : [],
            'historyBid'              : [],
            'historyAuctionResult'    : [],
            'historyImpressionResult' : [],
            'historyLeastWinningCost' : [],
        }


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


    def reset(self, period: int):
        self.remaining_budget = self.budget
        self.period_data = pd.read_parquet(self.DATA_PATH / f'bidding-period-{period}.parquet')

        self.max_timesteps: int = self.period_data.index.get_level_values('timeStepIndex').max()

        self.current_period = period
        self.current_timestep = 0

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


    def step(self, bids: Optional[NDArray] = None):
        if self.is_terminal():
            return self.get_obs(), 0, self.is_terminal(), self.get_info()

        if bids is None:
            bids = self.period_data.loc[self.current_timestep][self.advertiser_number, 'bid'].to_numpy()

        elif len(bids) != len(self.period_data.loc[self.current_timestep]):
            raise ValueError(f"Number of bids ({len(bids)}) does not match number of opportunities ({len(self.period_data.loc[self.current_timestep])})")

        pValues           = self.period_data.loc[self.current_timestep][self.advertiser_number, 'pValue'].to_numpy()
        pValuesSigmas     = self.period_data.loc[self.current_timestep][self.advertiser_number, 'pValueSigma'].to_numpy()
        leastWinningCosts = self.impressions.loc[self.current_period, self.current_timestep][(3, 'cost')].to_numpy() # type: ignore

        winning_bids, costs, conversions = self.simulate_ad_bidding(bids, pValues, pValuesSigmas, leastWinningCosts)

        if costs.sum() > self.remaining_budget - self.MIN_BUDGET:
            bid_mask = self.resolve_overcosted_bids(bids, costs)

            winning_bids = winning_bids * bid_mask
            costs        = costs        * bid_mask
            conversions  = conversions  * bid_mask

        self.remaining_budget -= costs.sum()
        reward = conversions.sum()

        # Update history
        self.history['historyPValueInfo'].append(np.stack([pValues, pValuesSigmas]).T)
        self.history['historyBid'].append(bids)
        self.history['historyAuctionResult'].append(np.stack([winning_bids, winning_bids, costs]).T)
        self.history['historyImpressionResult'].append(np.stack([conversions, conversions]).T)
        self.history['historyLeastWinningCost'].append(leastWinningCosts)

        # Running statistics
        self.num_conversions += reward
        self.num_wins += winning_bids.sum()

        self.current_timestep += 1

        return self.get_obs(), reward, self.is_terminal(), self.get_info()


    # Alternative (deterministic) implementation
    # def resolve_overcosted_bids(self, bids: NDArray, costs: NDArray):
    #     return costs.cumsum() <= self.remaining_budget


    def resolve_overcosted_bids(self, bids: NDArray, costs: NDArray):
        mask = np.ones_like(bids, dtype=bool)

        while bids[mask].sum() > self.remaining_budget:
            drop_ratio = 1 - self.remaining_budget / bids[mask].sum()

            drop_indices = np.random.choice(np.where(mask)[0], int(len(mask) * drop_ratio), replace=False)
            
            mask[drop_indices] = False
        
        return mask


    def simulate_ad_bidding(
            self,
            bids: NDArray,
            pValues: NDArray,
            pValueSigmas: NDArray,
            leastWinningCosts: NDArray,
        ):
        winning_bids = bids >= leastWinningCosts

        costs = leastWinningCosts * winning_bids

        sim_pValues = np.random.normal(loc=pValues, scale=pValueSigmas) * winning_bids
        sim_pValues = np.clip(sim_pValues, 0, 1)

        conversions = np.random.binomial(1, sim_pValues)

        return winning_bids, costs, conversions