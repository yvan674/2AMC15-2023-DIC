"""Greedy Agent.

Chooses the best scoring value with no thought about the future.
"""
import numpy as np
from gymnasium.spaces import Space

from dic import BaseAgent


class GreedyAgent(BaseAgent):
    def __init__(self, action_space: Space):
        """Chooses an action randomly unless there is something neighboring.

        Args:
            action_space: The possible action space received from the
                environment.
        """
        super().__init__()
        self.action_space = action_space

    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        raise NotImplementedError
