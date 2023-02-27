"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
import gymnasium as gym
from gymnasium.spaces import Space
import numpy as np

from dic import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, action_space: Space):
        """Agent that performs a random action every time.

        Args:
            action_space: The possible action space for the agent, received from
                the environment.
        """
        super().__init__()
        self.action_space = action_space

    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        return self.action_space.sample()
