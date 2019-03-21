import gym
import math
import os
import numpy as np

from gym import spaces
from utils import *

class TradingEnv(gym.Env):
    def __init__(self, train_data, init_capital, is_discrete, source):
        self.stock_return_rate_history = train_data
        self.n_stocks = train_data.shape[1] #should be only 2
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
        if self.source == 'M2': self.current_step = 2 #if mem 2 we need to to start at 2 for obs space
        else: self.current_step = 1
        self.stock_return_rate = self.stock_return_rate_history.loc[self.current_step]
        self.current_capital = self.init_capital
        return self._get_obs()

    def _get_obs(self):
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
        #previous capital is now capital before action
        prev_capital = self.current_capital
        #increment time step for data
        self.current_step += 1
        #return rate is return rate for that day
        self.stock_return_rate = self.stock_return_rate_history.loc[self.current_step]
        new_val, reward = self._invest(prev_capital, action)
        self.current_capital = new_val
        if self.current_step == self.n_steps - 1:
            done_flag = True
        else:
            done_flag = False
        return self._get_obs(), reward, done_flag

    def _invest(self, prev_capital, action):
        # cash for each stock
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
