"""Greedy Agent.

Chooses the best scoring value with no thought about the future.
"""
import numpy as np
from random import randint
from agents import BaseAgent


class ValueAgent(BaseAgent):
    def __init__(self, agent_number, observation, gamma, theta=0.001):
        """

        Args:
            agent_number: The index of the agent in the environment.
        """

        super().__init__(agent_number)
        rows, cols = observation.shape
        self.obs = observation
        self.gamma = gamma
        self.theta = theta

        self.states = [(i, j) for i in range(rows) for j in range(cols)
                       if self.obs[i, j] in (0, 3, 4)]
        print(len(self.states))
        self.values = {state: 0 for state in self.states}
        self.policy = self.value_iteration()

    def value_iteration(self) -> dict:
        i = 0
        delta = self.theta * 2
        while delta > self.theta:
            delta = 0
            for state in self.states:
                v = self.values[state]
                self.values[state] = self.max_value(state)
                delta = max(delta, abs(v - self.values[state]))
            
            print(f"Iteration {i}: delta = {delta}")
            i += 1
        
        return {state: self.best_action(state) for state in self.states}
 
    def max_value(self, state: tuple) -> float:
        new_values = []
        for action in (0, 1, 2, 3):
            total = 0
            for neighbor in neighbor_states(state):
                for reward in [0, 1]:
                    curr_prob = self.prob(state, action, neighbor, reward)
                    if curr_prob != 0:
                        total += curr_prob * (reward + self.gamma * self.values[neighbor])

            new_values.append(total)
        return max(new_values)
    
    def best_action(self, state: tuple) -> int:
        if self.obs[state] == 4:
            return 4
        
        best_value = 0
        best_action = None
        for action in (0, 1, 2, 3):
            total = 0
            for neighbor in neighbor_states(state):
                for reward in [0, 1]:
                    curr_prob = self.prob(state, action, neighbor, reward)
                    if curr_prob != 0:
                        total += curr_prob * (reward + self.gamma * self.values[neighbor])

            if total > best_value:
                best_value = total
                best_action = action
        
        return best_action
        
    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        x, y = info["agent_pos"][self.agent_number]
        # Check each neighboring cell if there is any dirt there
        return self.policy[(x, y)]


    def prob(self, state: tuple, action: int, new_state: tuple, reward: int) -> float:
        if self.obs[state] == 4:
            return float(0)
        
        neighbor = neighbor_states(state)[action]

        if self.obs[neighbor] == 0:
            return float(new_state == neighbor and reward == 0)
        if self.obs[neighbor] in (1, 2):
            return float(new_state == state and reward == 0)
        if self.obs[neighbor] == 3:
            return float(new_state == state and reward == 1)
        if self.obs[neighbor] == 4:
            return float(new_state == state and reward == 0)
        
        return float(0)

def neighbor_states(state: tuple) -> list:
    x, y = state
    # Actions 0, 1, 2, 3, 4
    return [(x, y+1), (x, y-1), (x-1, y), (x+1, y), (x, y)]