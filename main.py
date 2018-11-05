import numpy as np

from enviorment import TradingEnv
from gym import spaces
from Q_table import QLearningTable
from utils import *

episoides = 1000

def update():
    #by default the training is set to be 100 episodes per training
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # update env
            # did not finish TradingEnv for fresh env yet

            # RL choose action based on observation
            action = Q.choose_action(str(observation))

            # RL take action and get next observation and reward
            # return next step observation_, reward from env

            # RL learn from this transition
            Q.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break


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
    #Q = initialize_Q()
    Q = QLearningTable(actions=list(range(env.action_space.n)))

    print(Q.q_table)

    #number of episoides
    #episoides = 10

    #for _ in range(episoides):
        #state = env.reset()
        #state =
