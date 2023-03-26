"""Test Agent.

Cycles between each move for 200 steps each
"""
import numpy as np

from agents import BaseAgent


class TestAgent(BaseAgent):
    def __init__(self, agent_number: int):
        super().__init__(agent_number=agent_number)
        self.step = 0

    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        self.step += 1

        return (self.step % 1000) // 200
