"""Environment.

We define the grid environment for DIC in this file.
"""
import random
from time import time, sleep
from copy import deepcopy
from pathlib import Path
from warnings import warn

import numpy as np
from tqdm import trange

from dic import Grid, load_grid_file
from dic import EnvironmentGUI


class Environment:
    def __init__(self, grid_fp: Path,
                 headless: bool = False,
                 n_agents: int = 1,
                 agent_start_pos: list[tuple[int, int]] = None,
                 reward_fn: callable = None,
                 target_fps: int = 30):
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
                If None is provided, then a random start position for each
                agent is used.
            reward_fn: Custom reward function to use. It should have a
                signature of func(grid: Grid, info: dict) -> float. See the
                default reward function in this class for an example.
            target_fps: How fast the simulation should run if it is being shown
                in a GUI. This is a target, not the actual speed. If in
                headless mode, then the simulation will run as fast as
                possible. We may set a low FPS so we can actually see what's
                happening. Set to 0 or less to unlock FPS.
        """
        if not grid_fp.exists():
            raise FileNotFoundError(f"Grid {grid_fp} does not exist.")
        # Load the grid from the file
        self.grid_fp = grid_fp

        # Set up the environment as a blank state.
        self.grid = None
        self.headless = headless
        if target_fps <= 0:
            self.target_spf = 0.
        else:
            self.target_spf = 1. / target_fps
        self.gui = None

        # Set up initial agent positions
        self.n_agents = n_agents                 # Number of active agents
        self.agent_pos = None                    # Current agent positions
        self.agent_start_pos = agent_start_pos   # Where agents initially start
        self.agent_done = [False] * n_agents

        # Set up reward function
        if reward_fn is None:
            warn("No reward function provided. Using default reward.")
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = reward_fn
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()

        self.environment_ready = False
        self.reset()


    def _reset_info(self) -> dict:
        """Resets the info dictionary.

        info is a list of stats of the most recent step. It contains how many
        dirt tiles were cleaned, a list of if the agent moved, and a list of
        if an agent is done, i.e., succesfully moved onto the charger.
        agent_moved and agent_done are a boolean list.

        For example, if agent 0 moved, agent 1 failed to move, and agent 2
        moved, then agent_moved = [True, False, True].

        Same thing for agent_charging. agent_charging is True if the agent
        moved to the charger this turn

        Similarly, the index of dirt_cleaned is the number of dirt tiles
        cleaned by the agent at that index. If agent 0 cleaned 1 dirt tile,
        agent 1 cleaned 0 dirt tiles, and agent 2 cleaned 0 dirt tiles, then
        dirt_cleaned would be [1, 0, 0]
        """
        return {"dirt_cleaned": [0] * self.n_agents,
                "agent_moved": [False] * self.n_agents,
                "agent_charging": self.agent_done,
                "agent_pos": self.agent_pos}

    @staticmethod
    def _reset_world_stats() -> dict:
        return {"total_dirt_cleaned": 0,
                "total_steps": 0,
                "total_agent_moves": 0,
                "total_agents_at_charger": 0,
                "total_failed_moves": 0}

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
            self.agent_pos = deepcopy(self.agent_start_pos)
        else:
            # No positions were given. We place agents randomly.
            warn("No initial agent positions given. Randomly placing agents "
                 "on the grid.")
            for _ in range(self.n_agents):
                # First get all empty positions
                zeros = np.where(self.grid.cells == 0)
                idx = random.randint(0, len(zeros[0]) - 1)
                agent_pos.append((zeros[0][idx], zeros[1][idx]))
            self.agent_pos = agent_pos

    def get_observation(self) -> [np.ndarray, dict]:
        """Gets the current observation and information.

        Returns:
            - observation as an np.ndarray
            - info as a dict with keys ['dirt_cleaned', 'agent_moved',
              'agent_charging', 'agent_pos']
        """
        return self.grid.cells, self.info


    def reset(self, **kwargs) -> [np.ndarray, dict, dict]:
        """Reset the environment to an initial state.

        This is to reset the environment. You can fit it keyword arguments
        which will overwrite the initial arguments provided when initializing
        the environment.

        Example:
        >>> fp = Path("../grid_configs/base-room-1.grid")
        >>> e = Environment(fp, False, 1, None)
        >>> # Get the initial observation
        >>> observation, env_info = e.get_observation()
        >>> # Reset the environment, but for this training episode, we want
        >>> # to use 2 agents.
        >>> observation, env_info = e.reset(n_agents=2)

        Args:
            **kwargs: possible keyword options are the same as those for
                the environment initializer.
        Returns:
            - observation as an np.ndarray
            - info as a dict with keys ['dirt_cleaned', 'agent_moved',
                'agent_charging', 'agent_pos']
            - last run stats as a dict with keys ['total_dirt_cleaned',
                'total_steps', 'total_agent_moves', 'total_agents_at_charger',
                'total_failed_moves'].
        """
        world_stats = deepcopy(self.world_stats)
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
                case "target_fps":
                    self.target_spf = 1. / v
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
        self.world_stats = self._reset_world_stats()
        if not self.headless:
            self.gui = EnvironmentGUI(self.grid.cells.shape)
            self.gui.reset()
        else:
            if self.gui is not None:
                self.gui.close()

        self.environment_ready = True
        return self.grid.cells, self.info, world_stats

    def _move_agent(self, new_pos: tuple[int, int], agent_id: int):
        """Moves the agent, if possible.

        Args:
            new_pos: The new position of the agent.
            agent_id: The id of the agent. This is its index in the list of
                agent positions.
        """
        match self.grid.cells[new_pos]:
            case 0:  # Moved to an empty tile
                self.agent_pos[agent_id] = new_pos
                self.info["agent_moved"][agent_id] = True
                self.world_stats["total_agent_moves"] += 1
            case 1 | 2:  # Moved to a wall or obstacle
                self.world_stats["total_failed_moves"] += 1
                pass
            case 3:  # Moved to a dirt tile
                self.agent_pos[agent_id] = new_pos
                self.grid.cells[new_pos] = 0
                self.info["dirt_cleaned"][agent_id] += 1
                self.world_stats["total_dirt_cleaned"] += 1
                self.info["agent_moved"][agent_id] = True
                self.world_stats["total_agent_moves"] += 1
            case 4:  # Moved to the charger
                # Moving to charger is only permitted if the room is clean.
                # NOTE: This is a pending design decision.
                if self.grid.sum_dirt() == 0:
                    self.agent_pos[agent_id] = new_pos
                    self.agent_done[agent_id] = True
                    self.info["agent_charging"][agent_id] = True
                    self.world_stats["total_agents_at_charger"] += 1
                # Otherwise, the agent can't move and nothing happens
            case _:
                raise ValueError(f"Grid is badly formed. It has a value of "
                                 f"{self.grid.cells[new_pos]} at position "
                                 f"{new_pos}.")

    def step(self, actions: list[int]) -> [np.ndarray, float, bool, dict]:
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
            0) Current grid cells,
            1) The reward for the agent,
            2) If the terminal state has been reached, and
            3) State information.
        """
        self.world_stats["total_steps"] += 1
        if not self.headless:
            start_time = time()
        if not self.environment_ready:
            raise ValueError("reset() has not been called yet. "
                             "The environment still needs to be initialized.")
        # Verify that the number of actions and the number of agents is the
        # same
        if len(actions) != self.n_agents:
            raise ValueError(f"Number of actions provided is {len(actions)}, "
                             f"but the number of agents is {self.n_agents}.")

        self.info = self._reset_info()

        max_x = self.grid.n_cols - 1
        max_y = self.grid.n_rows - 1

        for i, action in enumerate(actions):
            if self.agent_done[i]:
                # The agent is already on the charger, so it is done.
                continue
            match action:
                case 0:  # Move down
                    new_pos = (self.agent_pos[i][0],
                               min(max_y, self.agent_pos[i][1] + 1))
                case 1:  # Move up
                    new_pos = (self.agent_pos[i][0],
                               max(0, self.agent_pos[i][1] - 1))
                    pass
                case 2:  # Move left
                    new_pos = (max(0, self.agent_pos[i][0] - 1),
                               self.agent_pos[i][1])
                    pass
                case 3:  # Move right
                    new_pos = (min(max_x, self.agent_pos[i][0] + 1),
                               self.agent_pos[i][1])
                    pass
                case 4:  # Stand still
                    new_pos = (self.agent_pos[i][0],
                               self.agent_pos[i][1])
                case _:
                    raise ValueError(f"Provided action {action} for agent {i} "
                                     f"is not one of the possible actions.")
            self._move_agent(new_pos, i)

        # Update the grid with the new agent positions and calculate the reward
        reward = self.reward_fn(self.grid, self.info)
        terminal_state = sum(self.agent_done) == self.n_agents
        if terminal_state:
            self.environment_ready = False

        if not self.headless:
            time_to_wait = self.target_spf - (time() - start_time)
            if time_to_wait > 0:
                sleep(time_to_wait)
            self.gui.render(self.grid.cells, self.agent_pos, self.info)

        return self.grid.cells, reward, terminal_state, self.info

    @staticmethod
    def _default_reward_function(grid: Grid, info: dict) -> float:
        """This is the default reward function.

        This is a very simple default reward function. It simply checks if any
        dirt tiles were cleaned during the step and provides a reward equal to
        the total number of dirt tiles cleaned.
        """
        return float(sum(info["dirt_cleaned"]))


if __name__ == '__main__':
    # This is testing code to load a single grid.
    base_grid_fp = Path("../grid_configs/base-room-1.grid")
    env = Environment(base_grid_fp, False, 1, target_fps=-1)
    obs, inf = env.reset()

    from agents.random_agent import RandomAgent
    agent = RandomAgent()

    for i in trange(1000):
        action = [agent.take_action(obs, inf)]
        obs, reward, terminal_state, inf = env.step(action)
        if terminal_state:
            break

    obs, inf = env.reset(headless=True)
    for i in trange(100000):
        action = [agent.take_action(obs, inf)]
        obs, reward, terminal_state, inf = env.step(action)
        if terminal_state:
            break

