import gym
import math
import os
import numpy as np

from gym import spaces
from gym.utils import seeding
from utils import *

class TradingEnv(gym.Env):
    def __init__(self, train_data, init_capital, is_discrete):
        self.stock_return_rate_history = train_data
        self.n_stocks = train_data.shape[1]
        self.n_steps = train_data.shape[0]

        self.init_capital = init_capital
        self.current_step = None
        self.stock_owned = None
        self.stock_return_rate = None
        self.current_capital = None
        self.is_discrete = is_discrete

        #actions are 0, 1, 2, ..., n indicating investing all money in instrument
        #0, 1, 2, ..., n respectivly for 1 time period
        self.action_space = spaces.Discrete(train_data.shape[1])
        #seed and reset
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return[seed]

    def _reset(self):
        self.current_step = 1
        #self.stock_owned = [0]*self.n_stocks
        self.stock_return_rate = self.stock_return_rate_history.loc[self.current_step]
        self.current_capital = self.init_capital
        return self._get_obs()

    def _get_obs(self):
        return_rate_list_temp = list(self.stock_return_rate)
        if not self.is_discrete:
            for i in range(len(return_rate_list_temp)):
                if return_rate_list_temp[i] < -0.01: return_rate_list_temp[i] = -1
                elif return_rate_list_temp[i] > 0.01: return_rate_list_temp[i] = 1
                else: return_rate_list_temp[i] = 0
        return return_rate_list_temp

    def _step(self, action):
        #previous capital is now capital before action
        prev_capital = self.current_capital
        #increment time step for data
        self.current_step += 1
        #return rate is return rate for that day
        self.stock_return_rate = self.stock_return_rate_history.loc[self.current_step]
        #new value is return rate of chosen stock times previous capital
        new_val = (self.stock_return_rate[action]+1) * prev_capital
        #current reward , log base 2 of new capital / init investment
        if new_val > prev_capital: reward = math.log((new_val-prev_capital),2)
        elif new_val == prev_capital: reward = 0
        else: reward = -math.log(abs(new_val - prev_capital),2)
        #reward = new_val/prev_capital
        self.current_capital = round(new_val)
        #done if on the last step, or we have doubled out investment
        # or self.current_capital >= 4*self.init_capital
        if self.current_step == self.n_steps - 1:
            done_flag = True
        else:
            done_flag = False
        return self._get_obs(), reward, done_flag
