"""Greedy Agent.

Chooses the best scoring value with no thought about the future.
"""
import numpy as np
from random import randint
from agents import BaseAgent


class GreedyAgent(BaseAgent):
    def __init__(self, agent_number):
        """Chooses an action randomly unless there is something neighboring.

        Args:
            agent_number: The index of the agent in the environment.
        """
        super().__init__(agent_number)

    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        x, y = info["agent_pos"][self.agent_number]
        # Check each neighboring cell if there is any dirt there
        if observation[x, y + 1] == 3:
            return 0
        elif observation[x, y - 1] == 3:
            return 1
        elif observation[x - 1, y] == 3:
            return 2
        elif observation[x + 1, y] == 3:
            return 3
        else:
            return randint(0, 4)
