"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

Difference for Deep Q is we use a nueral network to approximate our q function 
instead of using a table.  
"""

import numpy as np
import pandas as pd
import random

from model import q_nn


class QLearningTable:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # a list
        self.action_size = action_size
        self.memory = []
        self.lr = 0.01
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = q_nn(state_size, action_size)

    def choose_action(self, observation):
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose random action
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(self.action_size)
        else:
            # let the model predict
            action = self.model.predict(observation)
        return action

    def memory(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))

    def learn(self, state):
        #train network on batches
        batch_size = 32
        batch = random.sample(self.memory, batch_size)