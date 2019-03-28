import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from enviorment import TradingEnv
from Q_table import QLearningTable
from tabulate import tabulate
from utils import *

def update(env, Q):
    """
    The main method train the Q Learning agent.

    Parameters
    ----------
    env: TradingEnv Obj
        An instance the TradingEnv class parameratized to suit data
    Q: QLearningTable Obj
        An instance of the QLearningTable class parameratized to suit data
    """
    try: episodes = int(sys.argv[2])
    except: episodes = 10
    episodes = 10
    ending_cap = []
    # By default the training is set to be 100 episodes per training
    for episode in range(episodes):
        start = time.time()
        print('\n***Episoide Number*** ===>', episode)
        # Initial observation
        observation = env._reset()
        Q.reset_epsilon()
        done = False
        while not done:
            # Choose action based on observation
            action = Q.choose_action(env._get_obs())
            # Take action and get next observation and reward
            # Return next step observation_, reward from env
            observation_, reward, done = env._step(action)

            # Update Q table this transition
            Q.learn(str(observation), action, reward, str(observation_))

            # Swap observation
            observation = observation_

            # Break while loop when end of this episode
            if done:
                end = time.time()
                print('Completed episoide in ', end - start, ' secconds.\n')
                ending_cap.append(env.current_capital)
                break
    return

def test(test_env, Q):
    """
    Testing Q learning performance by observing policy performance over testing
    period. Plots capital obtained by following the best policy as per Q learning.

    Parameters
    ----------
    env: TradingEnv Obj
        An instance the TradingEnv class parameratized to suit data
    Q: QLearningTable Obj
        An instance of the QLearningTable class parameratized to suit data
    """
    observation = test_env._reset()
    test_cap = []
    done = False
    while not done:
        # Get action with maximum future expected reward
        action = action = Q.q_table.loc[str(test_env._get_obs())].idxmax()
        # Apply action
        observation_, reward, done = test_env._step(action)
        # Save current cash to observe over testing period
        test_cap.append(test_env.current_capital)
        # Current obs becomes next obs
        observation = observation_
        # If completed time frame
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
    # Must index at starting at 0
    train_data.index -= 1000
    # Get a 3 level uniform quantizer of training data for observations
    codebook, bounds = quantize(np.append(train_data['ibm'].values, train_data['msft'].values))
    # Init trading env
    obs_space = define_observations(n_stocks=2, options=[-1,0,1])
    env = TradingEnv(train_data, init_capital=1000, is_discrete=False, source='M')
    env.specify_quantization_ranges(bounds)
    # Init Q table
    Q = QLearningTable(actions=list(range(env.action_space_size)), observations=obs_space)
    Q.setup_table()
    # Train method
    update(env, Q)
    test_env = TradingEnv(test_data, init_capital=100, is_discrete=False, source='M')
    print(tabulate(Q.q_table, tablefmt="markdown", headers="keys"))
    max_actions = Q.q_table.idxmax(axis=1).values
    for i in Q.q_table.idxmax(axis=1).values:
        print('maximizing action', env.actions[i])
    test(test_env, Q)
    return

def iid_data():
    print('For IID Source')
    # Get train and test data for 5000 days where return rate is i.i.d
    train_data, test_data = split_data(create_iid(5000))
    test_data.index -= (train_data.shape[0] + test_data.shape[0])-1000
    # Init trading enviorment
    env = TradingEnv(train_data, init_capital=100, is_discrete = True, source='IID')
    # Init q learing table
    Q = QLearningTable(actions=list(range(env.action_space_size)), observations=[[0,0]])
    Q.setup_table()
    # Traing method
    update(env, Q)
    print(tabulate(Q.q_table, tablefmt="markdown", headers="keys"))
    test_env = TradingEnv(test_data, init_capital=100, is_discrete = True, source='IID')
    test(test_env, Q)
    return

def markov_data():
    print('For Markov Source')
    # Get train and test for 5000 days where return rates are dependent on previous day
    train_data, test_data = split_data(create_markov(5000))
    test_data.index -= (train_data.shape[0] + test_data.shape[0]) - 1000
    # Init trading enviourment
    env = TradingEnv(train_data, init_capital=10000, is_discrete=True, source='M')
    # Init q learning Q_table
    Q = QLearningTable(actions=list(range(env.action_space_size)), observations=train_data.drop_duplicates().values)
    Q.setup_table()

    # Training method
    update(env, Q)
    test_env = TradingEnv(test_data, init_capital=100, is_discrete=True, source='M')
    print(tabulate(Q.q_table, tablefmt="markdown", headers="keys"))
    test(test_env, Q)
    return

def markov_data2():
    print('For Markov Memory 2 Source')
    # Get train and test for 5000 days where return rates are dependent on previous day
    train_data, test_data = split_data(create_markov_memory_2(5000))
    test_data.index -= (train_data.shape[0] + test_data.shape[0]) - 1000
    # Init trading envioronment
    env = TradingEnv(train_data, init_capital=100, is_discrete=True, source='M2')
    # Init q learning Q_table
    Q = QLearningTable(actions=list(range(env.action_space_size)), observations=[[0,0]])
    # Training method
    update(env, Q)
    test_env = TradingEnv(test_data, init_capital=100, is_discrete=True, source='M2')
    print(tabulate(Q.q_table, tablefmt="markdown", headers="keys"))
    test(test_env, Q)
    return

def mix():
    print('For a mixture of Markov and IID Source')
    # Get train and test data for 5000 days
    train_data, test_data = split_data(create_markov_iid_mix(5000))
    test_data.index -= (train_data.shape[0] + test_data.shape[0]) - 1000
    # Init trading env, is not discrete for iid from np uniform module
    env = TradingEnv(train_data, init_capital=100, is_discrete=False, source='mix')
    # Init q learning table
    Q = QLearningTable(actions=list(range(env.action_space_size)))
    # Training method
    update(env, Q)
    test_env = TradingEnv(test_data, init_capital=100, is_discrete=False, source='mix')
    print(tabulate(Q.q_table, tablefmt="markdown", headers="keys"))
    test(test_env, Q)
    return

def train_markov_test_real():
    # Load real data and split into training and testing
    real_train_data, real_test_data = split_data(round_return_rate(get_data()))
    real_train_data.index -= 1000
    # Get codebook and bounds for a 3-level uniform quantizer
    codebook, bounds = quantize(real_train_data['msft'].values)
    # Compute an empirical transition matrix over training data with bounds
    # From 3 level uniform quantizer
    P = empirical_transition_matrix(real_train_data['msft'].values, bounds)
    # Generate markov data according to this transition matrix and codebook values
    train_data, test_data = split_data(create_custom_markov_samples(5000, codebook, P))
    # Define observation space
    obs_space = [[-1, 0], [0, 0], [1, 0]]
    # Enviorment for markov samples
    env = TradingEnv(train_data, init_capital=100000, is_discrete=False, source='M')
    # Set quantization ranges for bounds from 3 level uniform quantizer
    env.specify_quantization_ranges(bounds)
    Q = QLearningTable(actions=list(range(env.action_space_size)), observations=obs_space)
    Q.setup_table()
    # Train Q learning agent
    update(env, Q)
    # Create new trading enviorment for testing policy on real data
    test_env = TradingEnv(real_test_data, init_capital=100, is_discrete=False, source='Real')
    test_env.specify_quantization_ranges(bounds)
    print(tabulate(Q.q_table, tablefmt="markdown", headers="keys"))
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
    elif source_type == 'beta': train_markov_test_real()
    else: print('Invalid arguement.')
