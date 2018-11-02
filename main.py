import numpy as np

from enviorment import TradingEnv
from utils import *


if __name__ == '__main__':
    train_data = round_return_rate(get_data())
    env = TradingEnv(train_data, 100)
    print(train_data['ibm'].iloc[0:100:5])
    max, min = get_max_and_min(train_data)
    print('Max return rate', max, 'Min return rate', min)
    #print('Num Stocks', env.n_stocks)
    #print('Num Steps', env.n_steps)
    #print('Init Invest', env.init_capital)
    #print('Action Space', env.action_space)
    #print('Observation Space', env.observation_space)
