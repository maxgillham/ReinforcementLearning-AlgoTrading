"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
"""

import numpy as np
import pandas as pd

from numba import jit


class QLearningTable:
    def __init__(self, actions, observations):
        self.actions = actions  # a list
        self.lr = 0.01
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.observations = observations

    def setup_table(self):
        for obs in self.observations:
            self.q_table = self.q_table.append(
                pd.Series(
                [0]*len(self.actions),
                index=self.q_table.columns,
                name=str(list(obs))
                )
            )
        return


    @jit
    def choose_action(self, observation):
        #self.check_state_exist(observation)
        # choose "current best" action
        if np.random.rand() > self.epsilon: action = self.q_table.loc[str(observation)].idxmax()
        # choose random action
        else: action = np.random.choice(self.actions)
        #decay exploration rate if not below min exploration rate
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
        return action

    @jit
    def learn(self, s, a, r, s_):
        #self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        q_target = r + (self.gamma*self.q_table.loc[s_].max())
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    @jit
    def check_state_exist(self, state):
        if str(state) not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=str(state),
                )
            )
