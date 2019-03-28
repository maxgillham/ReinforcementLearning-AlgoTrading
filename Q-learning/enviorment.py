import math
import os
import numpy as np

from utils import *

class TradingEnv:
    def __init__(self, train_data, init_capital, is_discrete, source):
        """
        Initialize instance of the TradingEnv.

        Parameters
        ----------
        train_data: pandas.DataFrame
            Training set of daily return rates for a series of assets.
        init_capital: int
            The initial capital to begin episoides with.
        is_discrete: bool
            True if the return rates are quantized into a finite number of bins
            and False otherwise.
        source: str
            The type of source, or how the enviorment treats the source.  M for
            markov1, M2 for markov2, IID, Real
        """
        self.stock_return_rate_history = train_data
        self.n_stocks = train_data.shape[1]
        self.n_steps = train_data.shape[0]
        self.source = source
        self.init_capital = init_capital
        self.current_step = None
        self.stock_owned = None
        self.stock_return_rate = None
        self.current_capital = None
        self.is_discrete = is_discrete
        # Default quantization ranges
        self.quantization_ranges = [-0.01, 0.01]
        self.partition_ranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # Actions are combinations of partition values that sum to 1. ie for 3 stocks (.2, .3, .5)
        self.actions = define_actions(self.n_stocks, self.partition_ranges)
        self.action_space_size = len(self.actions)
        self._reset()

    def _reset(self):
        """
        Reset the capital to initial capital and current step. Used when a new
        episode begins.
        """
        # If mem 2 we need to to start at 2 for obs space
        if self.source == 'M2': self.current_step = 2
        else: self.current_step = 1
        self.stock_return_rate = self.stock_return_rate_history.loc[self.current_step]
        self.current_capital = self.init_capital
        return self._get_obs()

    def _get_obs(self):
        """
        Returns the current observation to be seen by the agent. If return rates
        are observed as markov, previous values are included in observation.

        Returns
        -------
        obs = list[]
            The observation at current time step.
        """
        if self.source == 'Real'  or self.source == 'M':
            return_rate_list_temp = list(self.stock_return_rate)
            if not self.is_discrete:
                for i in range(len(return_rate_list_temp)):
                    if return_rate_list_temp[i] < self.quantization_ranges[0]: return_rate_list_temp[i] = -1
                    elif return_rate_list_temp[i] > self.quantization_ranges[1]: return_rate_list_temp[i] = 1
                    else: return_rate_list_temp[i] = 0
            return return_rate_list_temp
        elif self.source =='M2':
            prev_2 = self.stock_return_rate_history.loc[self.current_step-1]
            prev_1 = self.stock_return_rate_history.loc[self.current_step]
            obs = list(prev_2)
            obs.extend(list(prev_1))
            for i in range(len(obs)):
                if obs[i] < self.quantization_ranges[0]: obs[i] = -1
                elif obs[i] > self.quantization_ranges[1]: obs[i] = 1
                else: obs[i] = 0
            return obs
        else:
            return [0,0]

    def _step(self, action):
        """
        Update current capital, current step, done flag and compute reward for
        each transition, given a control action.

        Parameters
        ----------
        action: int
            The index of the chosen control action in the list of feasible
            control actions

        Returns
        -------
        obs: list[]
            Observation after completing action
        reward: float
            Reward for action
        done_flag: bool
            Indicating if the end of training data is reached.
        """
        # Previous capital is now capital before action
        prev_capital = self.current_capital
        # Increment time step for data
        self.current_step += 1
        # Return rate is return rate for that day
        self.stock_return_rate = self.stock_return_rate_history.loc[self.current_step]
        new_val, reward = self._invest(prev_capital, action)
        self.current_capital = new_val
        if self.current_step == self.n_steps - 1:
            done_flag = True
        else:
            done_flag = False
        return self._get_obs(), reward, done_flag

    def _invest(self, prev_capital, action):
        """
        Invest capital according to the partition determined by the control action
        for 1 time step and compute reward.

        Parameters
        ----------
        prev_capital: float
            Previous capital before action.
        action: int
            Index of control action in list of actions

        Returns
        -------
        value: float
            Portfolio value after executing the previous action.
        reward: float
            Reward for executing action.
        """
        # Cash for each stock
        cash = []
        value = 0
        for i in self.actions[action]:
            cash.append(prev_capital * i)
        for j in range(self.n_stocks):
            value += (self.stock_return_rate[j]+1)*cash[j]
        reward = math.log(value, 2)
        return value, reward

    # Only neseisarry for non_discrete data - defaulted to -0.01, 0.0, 0.01
    def specify_quantization_ranges(self, ranges):
        self.quantization_ranges = ranges
        return
