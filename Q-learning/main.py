import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from enviorment import TradingEnv
from gym import spaces
from Q_table import QLearningTable

from utils import *

def update(env, Q):
    try: episodes = int(sys.argv[2])
    except: episodes = 10
    ending_cap = []
    #by default the training is set to be 100 episodes per training
    for episode in range(episodes):
        start = time.time()
        print('\n***Episoide Number*** ===>', episode)
        # initial observation
        observation = env._reset()

        done = False
        while not done:
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
                end = time.time()
                print('Completed episoide in ', end - start, ' secconds.\n')
                ending_cap.append(env.current_capital)
                break
    return

def test(test_env, Q):
    observation = test_env._reset()
    test_cap = []
    done = False
    while not done:

        #get expected reward for each action at this state
        state_action = Q.q_table.loc[str(test_env._get_obs()), :]

        # some actions may have the same expected reward, randomly choose on in these actions
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)

        observation_, reward, done = test_env._step(action)

        test_cap.append(test_env.current_capital)

        observation = observation_

        if done:
            break

    plt.scatter(np.arange(len(test_cap)), test_cap, marker='.', c='k' )

    plt.title('Capital Attained at Each Decision')
    plt.xlabel('Day')
    plt.ylabel('Capital Attained')
    plt.show()
    return

def real_data():
    print('For Real Data')
    train_data, test_data = split_data(round_return_rate(get_data()))
    #must index at starting at 0
    train_data.index -= 100
    #init trading env
    env = TradingEnv(train_data, init_capital=100, is_discrete=False)
    #init Q table
    Q = QLearningTable(actions=list(range(env.action_space_size)))
    #train method
    update(env, Q)
    test_env = TradingEnv(test_data, init_capital=100, is_discrete=False)
    print(Q.q_table)
    test(test_env, Q)
    return

def iid_data():
    print('For IID Source')
    #get train and test data for 5000 days where return rate is i.i.d
    train_data, test_data = split_data(create_iid(5000))
    test_data.index -= (train_data.shape[0] + test_data.shape[0])-100
    #train_data = pd.read_pickle("data/train_data_iid")
    #test_data = pd.read_pickle("data/test_data_iid")
    #init trading enviorment
    env = TradingEnv(train_data, init_capital=100, is_discrete = False)
    #init q learing table
    Q = QLearningTable(actions=list(range(env.action_space_size)))
    #traing method
    update(env, Q)
    #print(Q.q_table)
    test_env = TradingEnv(test_data, init_capital=100, is_discrete = False)
    test(test_env, Q)
    return

def markov_data():
    print('For Markov Source')
    #get train and test for 5000 days where return rates are dependent on previous day
    train_data, test_data = split_data(create_markov(5000))
    test_data.index -= (train_data.shape[0] + test_data.shape[0]) - 100
    #train_data = pd.read_pickle("data/train_data_mc")
    #test_data = pd.read_pickle("data/test_data_mc")
    #init trading envioourment
    env = TradingEnv(train_data, init_capital=100, is_discrete=True)
    #init q learning Q_table
    Q = QLearningTable(actions=list(range(env.action_space_size)))
    #training method
    update(env, Q)
    test_env = TradingEnv(test_data, init_capital=100, is_discrete=True)
    print(Q.q_table)
    test(test_env, Q)
    return

def markov_data2():
    print('For Markov Memory 2 Source')
    #get train and test for 5000 days where return rates are dependent on previous day
    train_data, test_data = split_data(create_markov_memory_2(5000))
    test_data.index -= (train_data.shape[0] + test_data.shape[0]) - 100
    #init trading envioourment
    env = TradingEnv(train_data, init_capital=100, is_discrete=True)
    #init q learning Q_table
    Q = QLearningTable(actions=list(range(env.action_space_size)))
    #training method
    update(env, Q)
    test_env = TradingEnv(test_data, init_capital=100, is_discrete=True)
    print(Q.q_table)
    test(test_env, Q)
    return

def mix():
    print('For a mixture of Markov and IID Source')
    #get train and test data for 5000 days
    train_data, test_data = split_data(create_markov_iid_mix(5000))
    test_data.index -= (train_data.shape[0] + test_data.shape[0]) - 100
    #init trading env, is not discrete for iid from np uniform module
    env = TradingEnv(train_data, init_capital=100, is_discrete=False)
    #init q learning table
    Q = QLearningTable(actions=list(range(env.action_space_size)))
    #training method
    update(env, Q)
    test_env = TradingEnv(test_data, init_capital=100, is_discrete=False)
    print(Q.q_table)
    test(test_env, Q)
    return

if __name__ == '__main__':
    try:
        source_type = sys.argv[1]
    except:
        print('\nMust pass arguement for source type. Arguement options are: \n1. markov\n2. markov2\n3. iid\n4. real'
              '\n5. mix')
        source_type = 'null'

    if source_type == 'markov': markov_data()
    elif source_type == 'markov2': markov_data2()
    elif source_type == 'iid': iid_data()
    elif source_type == 'real': real_data()
    elif source_type == 'mix': mix()
    else: print('Invalid arguement.')
