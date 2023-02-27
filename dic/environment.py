import random
from pathlib import Path
import ast
from warnings import warn

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Environment(gym.Env):
    def __init__(self, grid: Path, n_agents: int = 1,
                 agent_start_pos: list[tuple[int, int]] = None):
        """Creates the grid environment for the robot vacuum.

        Creates a Grid environment from the provided grid file. The number of
        agents is variable, allowing for multi-agent environments. If any
        start positions are provided, then the number of positions must be
        equal to the number of agents.

        Args:
            grid: Path to the grid file to use.
            n_agents: The number of agents to initialize
            agent_start_pos: List of tuples of where the agents should start.
                If None is provided, then a random start position for each agent
                is used.
        """
        # Set up the initial state of the environment
        self.grid_str = grid
        self.grid, self.n_goals = self._read_grid(grid)

        # Set up initial grid positions
        self.n_agents = n_agents
        self.agent_pos = None
        self.agent_start_pos = agent_start_pos
        self._initialize_agent_pos()

        # Define the observation space and actions space
        self.observation_space = gym.spaces.Box(low=-1, high=4,
                                                shape=self.grid.shape,
                                                dtype=np.int8)
        self.action_space = spaces.Tuple([spaces.Discrete(5)] * n_agents)

    def _initialize_agent_pos(self):
        """Initializes agent position from the givin initial variables.

        If None is given as the agent_start_pos, choose a position at random to
        place to the agent. If it is filled, try again until it finds a place
        that has not been filled yet. This assumes a mostly empty grid.
        """
        if self.agent_start_pos is not None:
            for i in range(self.n_agents):
                pos = (self.agent_start_pos[i][0], self.agent_start_pos[i][1])
                if self.grid[pos] != 0:
                    # making sure agents aren't placed on walls/chargers
                    raise ValueError(
                        "Attempted to place agent on top of wall or "
                        "charger")
                self.grid[pos] = 2
            self.agent_pos = np.array(self.agent_start_pos)
        else:
            agent_pos = []
            for i in range(self.n_agents):
                #
                agent_placed = False
                while not agent_placed:
                    pos = (random.randint(0, self.grid.shape[0]),
                           random.randint(0, self.grid.shape[1]))
                    if self.grid[pos] != 0:
                        continue
                    else:
                        self.grid[pos] = 2
                        agent_pos.append(pos)
                        agent_placed = True
            self.agent_pos = np.array(agent_pos)

    @staticmethod
    def _read_grid(file_name: Path) -> [np.ndarray, int]:
        """Opens and parses the grid text file."""
        with open(file_name) as f:
            lines = f.readlines()
        if len(lines) != 4:
            warn("a grid file should containt 4 lines (size, obstacles, goals, "
                 "charger). The current file contains more lines, but there "
                 "will be continued as if it is correct.")
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
        num_goals = len(goals)

        charger = lines[3].strip('charger = ')
        charger = ast.literal_eval(charger.strip('\n'))
        for x, y in charger:
            grid[x, y] = 3

        return grid, num_goals

    def reset(self, **kwargs):
        """Reset the environment to an initial state.

        Args:
            **kwargs: possible keyword options are the same as those for
                the environment initializer.
        """
        if not bool(kwargs):
            # Empty kwargs, so we just reset.
            self.grid = self._read_grid(self.grid_str)
            self.agent_pos = np.array(self.agent_start_pos)
        else:
            for k, v in kwargs.items():
                match k:
                    case "grid":
                        self.grid_str = v
                        self.grid, self.n_goals = self._read_grid(v)
                    case "n_agents":
                        self.n_agents = v
                        self.agent_pos = None
                        self.agent_start_pos = None
                        self._initialize_agent_pos()
                    case "agent_start_pos":
                        self.agent_start_pos = v
                        self._initialize_agent_pos()
                    case _:
                        raise ValueError(f"{k} is not one of the possible "
                                         f"keyword arguments.")

        return self.grid

    def step(self, actions: list[int]):
        """This function makes the agent take a step on the grid.

        Actions are provided as a list of integers. The integer values are:
            - 0: Move down
            - 1: Move up
            - 2: Move left
            - 3: Move right
            - 4: Stand still


        Args:
            actions: List of integers representing the action each agent should
                take.
        """
        def verify_agent_movement(new_pos: tuple) -> bool:
            """Verifies if a move is legal/possible."""
            grid_val = self.grid[new_pos]
            if grid_val == -1:
                # This would move us into a wall. Cancel the move.
                return False
            elif grid_val == 3 and self.n_goals != 0:
                # We can't finish before collecting all goalse
                return False
            else:
                return True

        # calculate the new positions of the agents
        for i in range(self.n_agents):
            if actions[i] == 0:  # Move down
                new_pos = self.agent_pos[i, 0] - 1, self.agent_pos[i, 1]
                if verify_agent_movement(new_pos):
                    self.grid[self.agent_pos[i, 0], self.agent_pos[i, 1]] = 0
                    self.agent_pos[i, 0] = max(0, self.agent_pos[i, 0] - 1)
                else:
                    continue
            elif actions[i] == 1:  # Move up
                new_pos = self.agent_pos[i, 0] + 1, self.agent_pos[i, 1]
                if verify_agent_movement(new_pos):
                    self.grid[self.agent_pos[i, 0], self.agent_pos[i, 1]] = 0
                    self.agent_pos[i, 0] = min(9, self.agent_pos[i, 0] + 1)
                else:
                    continue
            elif actions[i] == 2:  # Move left
                new_pos = self.agent_pos[i, 0], self.agent_pos[i, 1] - 1
                if verify_agent_movement(new_pos):
                    self.grid[self.agent_pos[i, 0], self.agent_pos[i, 1]] = 0
                    self.agent_pos[i, 1] = max(0, self.agent_pos[i, 1] - 1)
                else:
                    continue
            elif actions[i] == 3:  # Move right
                new_pos = self.agent_pos[i, 0], self.agent_pos[i, 1] + 1
                if verify_agent_movement(new_pos):
                    self.grid[self.agent_pos[i, 0], self.agent_pos[i, 1]] = 0
                    self.agent_pos[i, 1] = min(9, self.agent_pos[i, 1] + 1)
                else:
                    continue
            elif actions[i] == 4:  # Stand still
                continue

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

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
