import gym

from gym import spaces

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
        capital_range = [[0, 2*init_capital]]
        stock_return_rate_range = [[-5, 5]]
        self.observation_space = spaces.MultiDiscrete(stock_return_rate_range + capital_range)
