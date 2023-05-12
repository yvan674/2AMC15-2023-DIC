"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
import numpy as np

from agents import BaseAgent


class QLearnAgent(BaseAgent):
    def __init__(self, agent_number: int, gamma: float, theta=0.001, epsilon=0.05, alpha=0.001):
        """
        Set agent parameters.

        Args:
            agent_number: The index of the agent in the environment.
            gamma: loss rate.
            theta: minimal change.
            epsilon: epsilon greedy
            alpha: learning rate
        """
        super().__init__(agent_number)
        self.gamma = gamma
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = None

    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    # TODO: Dirt check function
    # TODO: Get new state to update Q table
    # TODO: Q learning implementation
    def take_action(self, observation: np.ndarray, info: None | dict) -> int:

        if self.Q is None:
            self.Q = np.zeros([observation.shape[0], observation.shape[1], 2 ** 4, 5])

        state = info['agent_pos']
        print(observation.shape[1])


        # take action according to epsilon greedy
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(self.Q[state, :])

        # compute reward
        reward = 1

        new_state = get_new_state(state)
        # down
        if action == 0:
            new_state = 1
        # up
        elif action == 1:
            new_state =

        # left
        elif action == 2:

        # right
        elif action == 3:

        else:
            new_state =


        # update the Q function
        self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])

        return action