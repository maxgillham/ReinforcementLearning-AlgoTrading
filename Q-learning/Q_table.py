import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, observations):
        """
        Initialize instance of the QLearningTable.

        Parameters
        ----------
        actions: list[int]
            The indicies of all feasible control actions defined in the TradingEnv.
        observations: list[]
            The set of all possible observations seen by the agent.
        """
        self.actions = actions
        self.lr = 0.001
        self.gamma = 0.85
        self.epsilon = 1
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.observations = observations

    def setup_table(self):
        """
        Setup Q table - initilize all state action pairs to be 0.
        Rows are possible states and columns are control actions.
        """
        for obs in self.observations:
            self.q_table = self.q_table.append(
                pd.Series(
                [0]*len(self.actions),
                index=self.q_table.columns,
                name=str(list(obs))
                )
            )
        return

    # Reset epsilon to 1
    def reset_epsilon(self):
        self.epsilon = 1
        return

    def choose_action(self, observation):
        """
        Choose a control action to persue for a given state.

        Parameters
        ----------
        observation: list
            The current observation.

        Returns
        -------
        actions: int
            The index in the set of actions list to execute.
        """
        # Choose "current best" action
        if np.random.rand() > self.epsilon: action = self.q_table.loc[str(observation)].idxmax()
        # Choose random action
        else: action = np.random.choice(self.actions)
        # Decay exploration rate if not below min exploration rate
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
        return action

    def learn(self, s, a, r, s_):
        """
        Update the Q table according to the Bellman Eqn for previous state, action
        reward and current state.

        Parameters
        ----------
        s: str
            The previos state.
        a: int
            The index of the previos action.
        r: float
            The reward for the previos state action pair.
        s_: str
            The (now) current state
        """
        #self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        q_target = r + (self.gamma*self.q_table.loc[s_].max())
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    # Depreciated.
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
