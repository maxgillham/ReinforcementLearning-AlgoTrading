import numpy as np

from enviorment import TradingEnv
from gym import spaces
from utils import *

episoides = 1000

def initialize_Q():
    Q = {}

    #each episoide is a state, making
    #into string for dictionary purpose
    states = []
    for i in range(episoides):
        states.append(str(i).zfill(3))

    for state in states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0
    return Q




if __name__ == '__main__':
    train_data = round_return_rate(get_data())
    env = TradingEnv(train_data, init_capital=100)
    max, min = get_max_and_min(train_data)
    #print('Max return rate', max, 'Min return rate', min)
    #print('Num Stocks', env.n_stocks)
    #print('Num Steps', env.n_steps)
    #print('Init Invest', env.init_capital)

    #init Q table
    #Q = np.zeros((env.observation_space.n, env.action_space.n))

    Q = initialize_Q()

    print(Q)
    
    #number of episoides
    #episoides = 10

    #for _ in range(episoides):
        #state = env.reset()
        #state =
