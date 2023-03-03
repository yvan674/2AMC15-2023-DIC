"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
import numpy as np


from dic import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self):
        """Agent that performs a random action every time. """
        super().__init__()

    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        return randint(0, 4)
