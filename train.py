"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.
"""
from argparse import ArgumentParser
from pathlib import Path

from tqdm import trange

from dic import Environment

# Add your agents here
from agents.null_agent import NullAgent
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent


def parse_args():
    p = ArgumentParser("DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--headless", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--out", type=Path,
                   help="Where to save training results.")

    return p.parse_args()


def main(grid_paths: list[Path], headless: bool, iters: int, out: Path):
    """Main loop of the program."""

    for grid in grid_paths:
        env = Environment(grid, headless, n_agents=1, agent_start_pos=None)
        obs, info = env.reset()
        agents = [NullAgent(),
                  GreedyAgent(env.action_space),
                  RandomAgent(env.action_space)]
        for agent in agents:
            for _ in trange(iters):
                action = agent.take_action(obs, info)
                obs, reward, terminated, truncated, info = env.step([action])

                if terminated:
                    obs, info = env.reset()


if __name__ == '__main__':
    args = parse_args()
