import gym
import numpy as np

from gym import spaces
from gym.utils import seeding
from utils import *

class TradingEnv(gym.Env):
    def __init__(self, train_data, init_capital):
        self.stock_return_rate_history = train_data
        self.n_stocks = train_data.shape[1]
        self.n_steps = train_data.shape[0]

        self.init_capital = init_capital
        self.current_step = None
        self.stock_owned = None
        self.stock_return_rate = None
        self.current_capital = None

        #5 options, 0%, 25%, 50%, 75% or 100% for each stock
        #actually just make actition 0 1 or 2, as invest all money in
        #IBM, invest all in Microsoft ext..
        self.action_space = spaces.Discrete(1*self.n_stocks)
        #observation space
        capital_range = [0, 2*init_capital]
        stock_return_rate_range = [0, get_max_and_min(self.stock_return_rate_history)[0]]
        #self.observation_space = spaces.Tuple((
            #spaces.Discrete(stock_return_rate_range),
            #spaces.Discrete(capital_range)))
        self.observation_space = spaces.MultiDiscrete([capital_range, stock_return_rate_range])
        #self.observation_space = spaces.Discrete(capital_range)

        #seed and reset
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return[seed]

    def _reset(self):
        self.current_step = 1
        #self.stock_owned = [0]*self.n_stocks
        self.stock_return_rate = self.stock_return_rate_history.iloc[self.current_step]
        self.current_capital = self.init_capital
        return self._get_obs()

    def _get_obs(self):
        obs = []
        #obs.extend(self.stock_owned)
        obs.extend(list(self.stock_return_rate))
        obs.append(self.current_capital)
        return obs

    def _step(self, action):
        #previous capital is now capital before action
        prev_capital = self.current_capital
        #new value is return rate of chosen stock times previous capital
        new_val = (self.stock_return_rate[action]+1) * prev_capital
        #current reward , needs oto be log
        reward = new_val - prev_capital
        #increment time step for data
        self.current_step += 1
        #return rate is return rate for that day
        self.stock_return_rate = self.stock_return_rate_history.iloc[self.current_step]
        self.current_capital = new_val
        done_flag = (self.current_step == self.n_steps - 1)
        return self._get_obs(), reward, done_flag
