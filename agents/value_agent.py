"""Greedy Agent.

Chooses the best scoring value with no thought about the future.
"""
from random import randint

import numpy as np

from agents import BaseAgent


class ValueAgent(BaseAgent):
    def __init__(self, agent_number: int, gamma: float, theta=0.001):
        """
        Set agent parameters.

        Args:
            agent_number: The index of the agent in the environment.
            gamma: loss rate.
            theta: minimal change.
        """
        super().__init__(agent_number)
        self.gamma = gamma
        self.theta = theta

    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """
        Default method which decides an action based on the environment.

        Args:
            observation: current environment.
            info: situation in current environment.

        Returns:
            int: action to be taken.
        """

        # Setup for VI
        self.obs = observation
        cols, rows = observation.shape
        # Col - row
        self.states = [
            (i, j)
            for i in range(cols)
            for j in range(rows)
            if observation[i, j] in (0, 3, 4)
        ]
        self.chargers = [
            (i, j) for i in range(cols) for j in range(rows) if observation[i, j] == 4
        ]

        # Initialize self.values based on situation in the room
        if 3 in self.obs:  # Still dirt to be cleaned
            self.values = {
                state: (1 if self.obs[state] == 3 else 0) for state in self.states
            }
            for charger in self.chargers:
                self.values[charger] = -1
            self.cleaning = True
        else:  # Go to charging station
            self.values = {
                state: (1 if state in self.chargers else 0) for state in self.states
            }
            self.cleaning = False

        self.value_iteration()

        # Use self.values to pick best action
        state = info["agent_pos"][self.agent_number]
        return self.generate_policy(state)

    def value_iteration(self):
        """
        Value iteration implementation. Keep optimizing V until change is
        smaller than theta. self.values is updated in place.
        """
        delta = 2 * self.theta  # Initial delta to ensure loop starts
        while delta > self.theta:
            delta = 0
            for state in self.states:
                old_value = self.values[state]
                self.values[state] = self.max_action(state, True)[0]
                delta = max(delta, abs(self.values[state] - old_value))

    def max_action(self, state: tuple[int, int]) -> tuple[float, int]:
        """
        Try all actions from the current state and determine the action with
        the highest expected value.

        Args:
            state: current state of the agent.

        Returns:
            tuple: value, action pair with the highest value.
        """
        value_action = []
        for action in (0, 1, 2, 3, 4):
            new_state = self.action_outcome(state, action)
            if new_state == (-1, -1):
                continue
            elif self.obs[new_state] == 3 and self.cleaning:
                value = 1
            elif self.obs[new_state] == 4 and not self.cleaning:
                value = 1
            elif self.obs[new_state] == 4 and self.cleaning:
                value = 0
            else:
                value = self.values[new_state] * self.gamma
            value_action.append((value, action))

        return max(value_action)

    def action_outcome(self, state, action) -> tuple:
        """
        Determines the result of a certain action when taken in a certain place.

        Args:
            state: current state of the agent.
            action: action to be taken.

        Returns:
            tuple: resulting state. (-1, -1) if the action is illegal.
        """
        # Col - row
        x, y = state
        match action:
            case 0:  # Down
                new_state = (x, y + 1)
            case 1:  # Up
                new_state = (x, y - 1)
            case 2:  # Left
                new_state = (x - 1, y)
            case 3:  # Right
                new_state = (x + 1, y)
            case 4:  # Stand still
                new_state = state
        if new_state in self.states:
            return new_state
        else:
            return (-1, -1)

    def generate_policy(self, state):
        return self.max_action(state)[1]
