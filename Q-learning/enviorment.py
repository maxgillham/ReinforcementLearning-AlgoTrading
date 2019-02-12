import gym
import math
import os
import numpy as np

from gym import spaces
from gym.utils import seeding
from scipy import misc
from utils import *

class TradingEnv(gym.Env):
    def __init__(self, train_data, init_capital, is_discrete, source):
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

        #actions are 0, 1, 2, ..., n indicating investing all money in instrument
        #0, 1, 2, ..., n respectivly for 1 time period
        self.action_space_size = int(train_data.shape[1] + misc.comb(train_data.shape[1], 2))
        self.action_space = spaces.Discrete(self.action_space_size)
        #seed and reset
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return[seed]

    def _reset(self):
        if self.source == 'M2': self.current_step = 2 #if mem 2 we need to to start at 2 for obs space
        else: self.current_step = 1
        #self.stock_owned = [0]*self.n_stocks
        self.stock_return_rate = self.stock_return_rate_history.loc[self.current_step]
        self.current_capital = self.init_capital
        return self._get_obs()

    def _get_obs(self):
        if self.source == 'Real'  or self.source == 'M':
            return_rate_list_temp = list(self.stock_return_rate)
            if not self.is_discrete:
                for i in range(len(return_rate_list_temp)):
                    if return_rate_list_temp[i] < -0.01: return_rate_list_temp[i] = -1
                    elif return_rate_list_temp[i] > 0.01: return_rate_list_temp[i] = 1
                    else: return_rate_list_temp[i] = 0
            return return_rate_list_temp
        elif self.source =='M2':
            return_rates = self.stock_return_rate_history.loc[self.current_step-1:self.current_step]
            return np.append(return_rates.values[0], return_rates.values[1])
        else:
            return [0,0,0,0]

    def _step(self, action):
        #previous capital is now capital before action
        prev_capital = self.current_capital
        #increment time step for data
        self.current_step += 1
        #return rate is return rate for that day
        self.stock_return_rate = self.stock_return_rate_history.loc[self.current_step]
        new_val, reward = self._invest(prev_capital, action)
        self.current_capital = round(new_val)
        if self.current_step == self.n_steps - 1:
            done_flag = True
        else:
            done_flag = False
        return self._get_obs(), reward, done_flag

    def _invest(self, prev_capital, action):
        #if investing all money in one stock
        if action in range(self.n_stocks): new_val = (self.stock_return_rate[action]+1) * prev_capital
        else:
            index_1 = action%self.n_stocks
            index_2 = index_1 - action//self.n_stocks
            new_val = (self.stock_return_rate[index_1]+1)*(prev_capital/2)
            new_val += (self.stock_return_rate[index_2]+1)*(prev_capital/2)
        #reward function for new value
        if new_val > prev_capital: reward = math.log((new_val-prev_capital),2)
        elif new_val == prev_capital: reward = 0
        else: reward = -math.log(abs(new_val - prev_capital),2)
        return new_val, reward

"""
This is some notes I took to figure out a way to map the discrete action space to
how the data is indexed


r0, r1, r2
[0, 0.5, 1]

0 [1,0,0] -return_rates(0)
1 [0,1,0] -return_rates(1)
2 [0,0,1] -return_rates(2)
3 [.5,0,0.5] -return_rates(0), return_rates(2)
4 [.5,.5,0] -return_rates(1), return_rates(0)
5 [0,.5,.5] - return_rates(1), return_rates(2)

n=3
i is action
if choice in range(n)
    return_rates(i)
else
    x = i mod n
    diff = floor(i/n)
    return_rates(x)
    return_rates(x-diff)
"""

"""
r0, r1, r2, r3
[0,0.5,1]

0 [1,0,0,0] - return_rates(0)
1 [0,1,0,0] - return_rates(1)
2 [0,0,1,0] - return_rates(2)
3 [0,0,0,1] - return_rates(3)
4 [.5,0,0,.5] - return_rates(0), return_rates(3)
5 [.5,.5,0,0] - return_rates(0), return_rates(1)
6 [0,.5,.5,0] - return_rates(1), return_rates(2)
7 [0,0,.5,.5] - return_rates(2), return_rates(3)
8 [.5,0,.5,0] - return_rates(0), return_rates(2)
9 [0,.5,0,.5] - return_rates(1), return_rates(3)

n=4
i is action
if choice in range(n)
    return_rates(i)
else
    x = i mod n
    diff = floor(i/n)
    return_rates(x)
    return_rates(x-diff)

"""
