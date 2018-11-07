import numpy as np

from enviorment import TradingEnv
from gym import spaces
from Q_table import QLearningTable
from utils import *

episoides = 1000

def update():
    #by default the training is set to be 100 episodes per training
    for episode in range(5):
        # initial observation
        env._reset()

        while True:
            # update env
            # did not finish TradingEnv for fresh env yet

            # RL choose action based on observation
            action = Q.choose_action(env._get_obs())
            print('action', action)
            print('with capital', env.current_capital)
            #observation = env._get_obs()

            # RL take action and get next observation and reward
            # return next step observation_, reward from env
            observation_, reward, done = env._step(action)

            print('reward', reward)
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

    #init Q table
    Q = QLearningTable(actions=list(range(env.action_space.n)))

    #print(Q.q_table)
    #print(Q.choose_action(env._get_obs()))

    update()
    
   