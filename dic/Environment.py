import gym
import numpy as np
import ast


def read_grid(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    if len(lines) != 4:
        print('a grid file should containt 4 lines (size, obstacles, goals, charger)')
        print('the current file contains more lines, but there will be continued as if it is correct')
    size = lines[0].strip('size = ')
    size = ast.literal_eval(size.strip('\n'))
    grid = np.zeros(size)

    obstacles = lines[1].strip('obstacles = ')
    obstacles = ast.literal_eval(obstacles.strip('\n'))
    for x1, y1, x2, y2 in obstacles:
        for i in range(x1, x2 + 1):
            for j in range(y1, y2 + 1):
                grid[i, j] = -1

    goals = lines[2].strip('goals = ')
    goals = ast.literal_eval(goals.strip('\n'))
    for x, y in goals:
        grid[x, y] = 1
    g = len(goals)

    charger = lines[3].strip('charger = ')
    charger = ast.literal_eval(charger.strip('\n'))
    for x, y in charger:
        grid[x, y] = 3

    return grid, g

class Environment(gym.Env):
    def __init__(self, grid: str, n_agents=1, agent_start_pos=[(1,1)]):

        # Set up the initial state of the environment
        self.grid_str = grid
        self.grid, self.n_goals = read_grid(grid)
        self.agent_start_pos = agent_start_pos

        # Define the observation space and action space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10, 10), dtype=np.int8)
        self.action_space = gym.spaces.Discrete(5)

        # Create the agents and set their positions in the grid
        self.agent_pos = np.array(agent_start_pos)
        self.n_agents = n_agents
        for i in range(self.n_agents):
            pos = (self.agent_pos[i, 0], self.agent_pos[i, 1])
            if self.grid[pos] == -1 or self.grid[pos] == 3:  # making sure agents aren't placed on walls/chargers
                raise Exception("Attempted to place agent on top of wall or charger")
            self.grid[pos] = 2

    def reset(self):
        """Reset the environment to its initial state"""
        self.grid = read_grid(self.grid_str)
        self.agent_pos = np.array(self.agent_start_pos)
        return self.grid

    def step(self, action):
        """This function makes the agent take a step on the grid"""

        if type(action) == int:  # for when the user has only 1 agent, and wants to pass an integer as action
            action = [action]

        # Remove the agents from their previous location
        for i in range(self.n_agents):
            self.grid[self.agent_pos[i, 0], self.agent_pos[i, 1]] = 0

        # calculate the new positions of the agents
        for i in range(self.n_agents):
            if action[i] == 0:  # Move down
                val = self.grid[self.agent_pos[i, 0] - 1, self.agent_pos[i, 1]]
                if val == -1:  # if we would move onto a wall, we don't move
                    continue
                elif val == 3 and self.n_goals != 0:  # we can not finish if we haven't collected all goals
                    continue
                self.agent_pos[i, 0] = max(0, self.agent_pos[i, 0] - 1)
            elif action[i] == 1:  # Move up
                val = self.grid[self.agent_pos[i, 0] + 1, self.agent_pos[i, 1]]
                if val == -1:  # if we would move onto a wall, we don't move
                    continue
                elif val == 3 and self.n_goals != 0:  # we can not finish if we haven't collected all goals
                    continue
                self.agent_pos[i, 0] = min(9, self.agent_pos[i, 0] + 1)
            elif action[i] == 2:  # Move left
                val = self.grid[self.agent_pos[i, 0], self.agent_pos[i, 1] - 1]
                if val == -1:  # if we would move onto a wall, we don't move
                    continue
                elif val == 3 and self.n_goals != 0:  # we can not finish if we haven't collected all goals
                    continue
                self.agent_pos[i, 1] = max(0, self.agent_pos[i, 1] - 1)
            elif action[i] == 3:  # Move right
                val = self.grid[self.agent_pos[i, 0], self.agent_pos[i, 1] + 1]
                if val == -1:  # if we would move onto a wall, we don't move
                    continue
                elif val == 3 and self.n_goals != 0:  # we can not finish if we haven't collected all goals
                    continue
                self.agent_pos[i, 1] = min(9, self.agent_pos[i, 1] + 1)
            elif action[i] == 4:  # Stand still
                pass


        # Update the grid with the new agent positions and calculate the reward
        reward = 0
        delbots = []
        for i in range(self.n_agents):
            r = self.grid[self.agent_pos[i, 0], self.agent_pos[i, 1]]
            reward += r
            if r == 1:
                self.n_goals -= 1
            elif r == 3 and self.n_goals == 0:
                delbots.append(i)
                continue
            self.grid[self.agent_pos[i, 0], self.agent_pos[i, 1]] = 2

        for i in delbots:  # if any
            self.n_agents -= 1
            if self.n_agents == 0:  # TERMINAL STATE
                return self.grid, reward, True, {}
            self.agent_pos = np.delete(self.agent_pos, i, 0)

        return self.grid, reward, False, {}




