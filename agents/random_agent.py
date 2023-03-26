"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
import numpy as np

from agents import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that performs a random action every time. """
    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        return randint(0, 4)
