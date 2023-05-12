"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
import numpy as np
import math
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
        self.dirtGrid = np.zeros(4)

    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    def reward_func(self, observation, state):
        if observation[state[0], state[1]] in [1, 2]:
            return -1000

        if state[2] != 15:
            if observation[state[0], state[1]] == 3:
                return 5
            else:
                return 0
        else:
            if observation[state[0], state[1]] == 4:
                return 50
            else:
                return 0

    def get_new_pos(self, observation, action, state):
        action_map = {0: [state[0], state[1] - 1, state[2]],  # down
                      1: [state[0], state[1] + 1, state[2]],  # up
                      2: [state[0] - 1, state[1], state[2]],  # left
                      3: [state[0], state[1] + 1, state[2]],  # right
                      }
        new_state = action_map.get(action, state)

        if observation[new_state[0], new_state[1]] in [1, 2]:
            return state
        else:
            return new_state

    # TODO: Dirt check function
    def dirt_function(self, observation: np.ndarray, state):
        #check which quarter
        height = observation.shape(0)
        width = observation.shape(1)
        if state[0] < height/2:
            if state[1] < width/2:
                quarter = 0
            else:
                quarter = 1
        else:
            if state[1] < width/2:
                quarter = 2
            else:
                quarter = 3

        #check if already dirt free
        if self.dirtGrid(quarter) == 1:
            return dirt_byte_converter(self.dirtGrid)
        
        dirty = False



        if quarter == 1:
            for i in range(0, math.floor(height/2)):
                for j in range(0, math.floor(width/2)):
                    if observation[i][j] == 3:
                        dirty = True
                        break
                if dirty:
                    break

        
        if quarter == 2:
            for i in range(0, math.floor(height/2)):
                for j in range(math.ceil(width/2), width):
                    if observation[i][j] == 3:
                        dirty = True
                        break
                if dirty:
                    break

        if quarter == 3:
            for i in range(math.ceil(height/2), height):
                for j in range(0, math.floor(width/2)):
                    if observation[i][j] == 3:
                        dirty = True
                        break
                if dirty:
                    break
                
        if quarter == 4:
            for i in range(math.ceil(height/2), height):
                for j in range(math.ceil(width/2), width):
                    if observation[i][j] == 3:
                        dirty = True
                        break
                if dirty:
                    break
        
        if dirty == False:
            self.dirtGrid[quarter] = 1
        return dirt_byte_converter(self.dirtGrid)

        #check if there is dirt
        #update dirt 
        
    def dirt_byte_converter(dirtGrid):
        number = 0
        if dirtGrid[0] == 1:
            number += 1
        if dirtGrid[1] == 1:
            number += 2
        if dirtGrid[2] == 1:
            number += 4
        if dirtGrid[3] == 1:
            number += 8
        return number

    # TODO: Get new state to update Q table
    # TODO: Q learning implementation
    def take_action(self, observation: np.ndarray, info: None | dict) -> int:

        if self.Q is None:
            self.Q = np.zeros([observation.shape[0], observation.shape[1], 2 ** 4, 4])
            self.Q[:, 0, :, :] = -10000  # first column we don't want to visit
            self.Q[:, -1, :, :] = -10000  # last column we don't want to visit
            self.Q[0, :, :, :] = -10000  # first row we don't want to visit
            self.Q[-1, :, :, :] = -10000  # last row we don't want to visit

        state = info['agent_pos']
        print(observation.shape[1])


        # take action according to epsilon greedy
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, 4)
            if state[0]== 1 and action == 0:
        else:
            action = np.argmax(self.Q[state[0], state[1], state[2], :])

        # new state
        new_state = self.get_new_pos(observation, action, state)

        # compute reward
        reward = self.reward_func(observation, new_state)

        # update the Q function
        self.Q[state[0], state[1], state[2], action] += \
            self.alpha * (reward + self.gamma * np.max(self.Q[new_state[0], new_state[1], new_state[2], :]) -
            self.Q[state[0], state[1], state[2], action])

        return action


