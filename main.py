import numpy as np
import matplotlib.pyplot as plt

from enviorment import TradingEnv
from gym import spaces
from Q_table import QLearningTable
from utils import *

episodes = 100

def update():
    episode_num = []
    ending_cap = []
    #by default the training is set to be 100 episodes per training
    for episode in range(episodes):
        print('Episoide ', episode)
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
            Q.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                print('Completed episoide...\nEnding episoide with: $', env.current_capital)
                break
    plt.scatter(episode_num, ending_cap, marker='.', c='k' )
    plt.title('Capital Attained at Each Episode')
    plt.xlabel('Episode')
    plt.ylabel('Capital Attained')
    plt.show()
    return


if __name__ == '__main__':
    train_data = round_return_rate(get_data())
    env = TradingEnv(train_data, init_capital=100)

    #init Q table
    Q = QLearningTable(actions=list(range(env.action_space.n)))

    #print(Q.q_table)
    #print(Q.choose_action(env._get_obs()))

    update()
