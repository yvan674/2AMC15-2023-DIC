"""Null Agent.

An agent which does nothing.
"""
import numpy as np

from dic import BaseAgent


class NullAgent(BaseAgent):
    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        return 4
