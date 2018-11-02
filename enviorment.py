import gym

from gym import spaces
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
        #actually just do buy sell hold for now
        self.action_space = spaces.Discrete(3*self.n_stocks)
        #observation space
        capital_range = 2*init_capital
        stock_return_rate_range = get_max_and_min(self.stock_return_rate_history)[0]
        #self.observation_space = spaces.Tuple((
            #spaces.Discrete(stock_return_rate_range),
            #spaces.Discrete(capital_range)))
        #self.observation_space = spaces.MultiDiscrete(capital_range + stock_return_rate_range)
        self.observation_space = spaces.Discrete(capital_range)

        #self._seed()
        #self._reset()
