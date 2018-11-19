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


class deepQ:
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
            action = np.random.choice(self.action_size)
            print('random action', action)
        else:
            # let the model predict
            pred_unformat = self.model.predict(np.array([observation]))
            action = np.argmax(pred_unformat[0])
            print('nn action', action)
        return action

    def remember(self, s, a, r, s_, d):
        self.memory.append((s, a, r, s_, d))

    def learn(self):
        #train network on batches
        batch_size = 32
        #randomly choose 32 samples from previously seen data
        batch = random.sample(self.memory, batch_size)


        s = np.array([m[0] for m in batch])
        a = np.array([m[1] for m in batch])
        r = np.array([m[2] for m in batch])
        s_ = np.array([m[3] for m in batch])
        d = np.array([m[4] for m in batch])


        #Q(s_,a)
        target = r + self.gamma * np.argmax(self.model.predict(s_), axis=1)
        #target[d] = r[d]
        #Q(s,a)
        target_ = self.model.predict(s)

        target_[range(batch_size), a] = target

        self.model.fit(s, target_, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #default name to override everytime
    def save_model(self):
        self.model.save_weights('train.h5')

    def load_model(self):
        self.model.load_weights('train.h5')

    def test_pred(self, observation):
        pred_unformat = self.model.predict(np.array([observation]))
        return np.argmax(pred_unformat[0])
