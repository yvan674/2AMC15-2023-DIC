"""Environment.

We define the grid environment for DIC in this file.
"""
import random
from pathlib import Path
from warnings import warn

import numpy as np

from dic import Grid, load_grid_file


class Environment:
    def __init__(self, grid_fp: Path,
                 headless: bool = False,
                 n_agents: int = 1,
                 agent_start_pos: list[tuple[int, int]] = None,
                 reward_fn: callable = None):
        """Creates the grid environment for the robot vacuum.

        Creates a Grid environment from the provided grid file. The number of
        agents is variable, allowing for multi-agent environments. If any
        start positions are provided, then the number of positions must be
        equal to the number of agents.

        This environment follows the general principles of reinforcment
        learning. It can be though of as a function E : action -> observation
        where E is the environment represented as a function.

        Args:
            grid_fp: Path to the grid file to use.
            n_agents: The number of agents this environment should support.
            agent_start_pos: List of tuples of where each agent should start.
                If None is provided, then a random start position for each agent
                is used.
            reward_fn: Custom reward function to use. It should have a
                signature of func(grid: Grid, info: dict) -> float. See the
                default reward function in this class for an example.
        """
        if not grid_fp.exists():
            raise FileNotFoundError(f"Grid {grid_fp} does not exist.")
        # Load the grid from the file
        self.grid_fp = grid_fp

        # Set up the environment as a blank state.
        self.grid = None
        self.headless = headless

        # Set up initial agent positions
        self.n_agents = n_agents                 # Number of active agents
        self.agent_pos = None                    # Current agent positions
        self.agent_start_pos = agent_start_pos   # Where agents initially start

        if reward_fn is None:
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = reward_fn

        self.info = self._reset_info()

        self.environment_ready = False

    def _reset_info(self):
        """Resets the info dictionary.

        info is a list of stats of the most recent step. It contains how many
        dirt tiles were cleaned, a list of if the agent moved, and a list of
        if an agent is done, i.e., succesfully moved onto the charger.
        agent_moved and agent_done are a boolean list.

        For example, if agent 0 moved, agent 1 failed to move, and agent 2
        moved, then agent_moved = [True, False, True].

        Same thing for agent_done. agent_done is True once the agent has
        moved to the charger.

        Similarly, the index of dirt_cleaned is the number of dirt tiles cleaned
        by the agent at that index. If agent 0 cleaned 1 dirt tile, agent 1
        cleaned 0 dirt tiles, and agent 2 cleaned 0 dirt tiles, then
        dirt_cleaned would be [1, 0, 0]

        """
        return {"dirt_cleaned": [0] * self.n_agents,
                "agent_moved": [False] * self.n_agents,
                "agent_done": [False] * self.n_agents}


    def _initialize_agent_pos(self):
        """Initializes agent position from the givin initial variables.

        If None is given as the agent_start_pos, choose a position at random to
        place to the agent. If it is filled, try again until it finds a place
        that has not been filled yet. This assumes a mostly empty grid.
        """
        agent_pos = []
        if self.agent_start_pos is not None:
            # We try placing each agent at every requested position.
            for i in range(self.n_agents):
                pos = (self.agent_start_pos[i][0], self.agent_start_pos[i][1])
                if self.grid.cells[pos] == 0:
                    # Cell is empty. We can place the agent there.
                    agent_pos.append(pos)
                else:
                    # Agent is placed on walls/obstacle/dirt/charger
                    raise ValueError(
                        "Attempted to place agent on top of wall or "
                        "charger")
            self.agent_pos = np.array(self.agent_start_pos)
        else:
            # No positions were given. We place agents randomly.
            for i in range(self.n_agents):
                # First get all empty positions
                zeros = np.where(self.grid.cells == 0)
                idx = random.randint(0, len(zeros[0]))
                agent_pos.append((zeros[0][idx], zeros[1][idx]))
            self.agent_pos = np.array(agent_pos)

    # @staticmethod
    # def _read_grid(file_name: Path) -> [np.ndarray, int]:
    #     """Opens and parses the grid text file."""
    #     with open(file_name) as f:
    #         lines = f.readlines()
    #     if len(lines) != 4:
    #         warn("a grid file should containt 4 lines (size, obstacles, goals, "
    #              "charger). The current file contains more lines, but there "
    #              "will be continued as if it is correct.")
    #     size = lines[0].strip('size = ')
    #     size = ast.literal_eval(size.strip('\n'))
    #     grid = np.zeros(size)
    #
    #     obstacles = lines[1].strip('obstacles = ')
    #     obstacles = ast.literal_eval(obstacles.strip('\n'))
    #     for x1, y1, x2, y2 in obstacles:
    #         for i in range(x1, x2 + 1):
    #             for j in range(y1, y2 + 1):
    #                 grid[i, j] = -1
    #
    #     goals = lines[2].strip('goals = ')
    #     goals = ast.literal_eval(goals.strip('\n'))
    #     for x, y in goals:
    #         grid[x, y] = 1
    #     num_goals = len(goals)
    #
    #     charger = lines[3].strip('charger = ')
    #     charger = ast.literal_eval(charger.strip('\n'))
    #     for x, y in charger:
    #         grid[x, y] = 3
    #
    #     return grid, num_goals

    def reset(self, **kwargs) -> [np.ndarray, dict]:
        """Reset the environment to an initial state.

        This is to reset the environment. You can fit it keyword arguments
        which will overwrite the initial arguments provided when initializing
        the environment.

        Example:
        >>> fp = Path("../grid_configs/base-room-1.grid")
        >>> e = Environment(fp, False, 1, None)
        >>> # Do initial reset to initialize the environment
        >>> observation, env_info = e.reset()
        >>> # Reset the environment, but for this training episode, we want
        >>> # to use 2 agents.
        >>> observation, env_info = e.reset(n_agents=2)

        Args:
            **kwargs: possible keyword options are the same as those for
                the environment initializer.
        """
        for k, v in kwargs.items():
            # Go through each possible keyword argument.
            match k:
                case "grid_fp":
                    self.grid_fp = v
                case "n_agents":
                    self.n_agents = v
                    self.agent_pos = None
                    self.agent_start_pos = None
                case "agent_start_pos":
                    self.agent_start_pos = v
                case "headless":
                    self.headless = v
                case _:
                    raise ValueError(f"{k} is not one of the possible "
                                     f"keyword arguments.")

        if self.agent_start_pos is not None:
            if len(self.agent_start_pos) != self.n_agents:
                raise ValueError(f"Number of agents {self.n_agents} does not "
                                 f"agree with number of starting positions "
                                 f"{len(self.agent_start_pos)}.")

        self.grid = load_grid_file(self.grid_fp)
        self._initialize_agent_pos()
        self.info = self._reset_info()

        self.environment_ready = True
        return self.grid.cells, self.info

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
                take. The index of the action corresponds to which agent did
                which action.

        Returns:

        """
        self.info = self._reset_info()

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

        return self.grid.cells, reward, False, self.info

    @staticmethod
    def _default_reward_function(grid: Grid, info: dict) -> float:
        """This is the default reward function.

        This is a very simple default reward function. It simply checks if any
        dirt tiles were cleaned during the step and provides a reward equal to
        the total number of dirt tiles cleaned.
        """
        return float(sum(info["dirt_cleaned"]))

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


if __name__ == '__main__':
    # This is testing code to load a single grid.
    base_grid_fp = Path("../grid_configs/base-room-1.grid")
    env = Environment(base_grid_fp, False, 1, None)
    obs, inf = env.reset()
    breakpoint()
