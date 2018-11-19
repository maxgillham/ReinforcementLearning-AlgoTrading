import time
import os
import numpy as np
import matplotlib.pyplot as plt

from enviorment import TradingEnv
from gym import spaces
from deep_q_agent import deepQ

from utils import *

episodes = 1

def train():
    #by default the training is set to be 100 episodes per training
    for episode in range(episodes):
        start = time.time()
        print('\n***Episoide Number*** ===>', episode)
        # initial observation
        observation = env._reset()

        done = False
        while not done:
            # update env
            # did not finish TradingEnv for fresh env yet

            # RL choose action based on observation
            action = Q.choose_action(env._get_obs())

            # RL take action and get next observation and reward
            # return next step observation_, reward from env
            observation_, reward, done = env._step(action)

            # RL learn from this transition
            Q.remember(observation, action, reward, observation_, done)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                end = time.time()
                print('Completed episoide in ', end - start, ' secconds.\nFinal portfolio value: $', env.current_capital)
                break
            if len(Q.memory) > 32:
                Q.learn()
    return

def test():
    step_num = []
    portfolio_val = []

    observation = test_env._reset()
    done = False

    for i in range(test_env.n_steps):

        action = test_Q.test_pred(observation)
        print(action)
        observation_, reward, done = test_env._step(action)
        observation = observation_
        step_num.append(test_env.current_step)
        portfolio_val.append(test_env.current_capital)
        if done:
            print('ending at:', test_env.current_capital)
            break
    plt.scatter(step_num, portfolio_val, marker='.', c='k')
    plt.title('Portfolio Value at Each Step of Test Data')
    plt.xlabel('Day Number')
    plt.ylabel('Portfolio Value')
    plt.show()
    return



if __name__ == '__main__':

    train_data, test_data = split_data(round_return_rate(get_data()))
    #must index at starting at 0
    train_data.index -= 1000
    #init trading env
    env = TradingEnv(train_data, init_capital=1000)

    #init deep q agent
    Q = deepQ(state_size=len(env._get_obs()), action_size=env.action_space.n)

    #train method
    train()

    Q.save_model()

    test_env = TradingEnv(test_data, init_capital=1000)
    test_Q = deepQ(state_size=len(test_env._get_obs()), action_size=env.action_space.n)
    test_Q.load_model()
    test()
