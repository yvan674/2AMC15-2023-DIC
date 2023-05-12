"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.

In this example training script, we use command line arguments. Feel free to
change this to however you want it to work.
"""
from argparse import ArgumentParser
from pathlib import Path

from tqdm import trange

try:
    from world import Environment

    # Add your agents here
    from agents.null_agent import NullAgent
    from agents.greedy_agent import GreedyAgent
    from agents.random_agent import RandomAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.extend(root_path)

    from world import Environment

    # Add your agents here
    from agents.null_agent import NullAgent
    from agents.greedy_agent import GreedyAgent
    from agents.random_agent import RandomAgent


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--out", type=Path, default=Path("results/"),
                   help="Where to save training results.")

    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, out: Path, random_seed: int):
    """Main loop of the program."""

    for grid in grid_paths:
        # Set up the environment and reset it to its initial state
        env = Environment(grid, no_gui, n_agents=1, agent_start_pos=None,
                          sigma=sigma, target_fps=fps, random_seed=random_seed,
                          reward_fn='custom')
        obs, info = env.get_observation()

        # Set up the agents from scratch for every grid
        # Add your agents here
        # agents = [NullAgent(0),
        #           GreedyAgent(0),
        #           RandomAgent(0)]

        agents = [GreedyAgent(0)]

        # Iterate through each agent for `iters` iterations
        for agent in agents:
            for _ in trange(iters):
                # Agent takes an action based on the latest observation and info
                action = agent.take_action(obs, info)

                # The action is performed in the environment
                obs, reward, terminated, info = env.step([action])

                # If the agent is terminated, we reset the env.
                if terminated:
                    obs, info, world_stats = env.reset()
                agent.process_reward(obs, reward)
            obs, info, world_stats = env.reset()
            print(world_stats)

            Environment.evaluate_agent(grid, [agent], 1000, out, 0.2)



if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.out,
         args.random_seed)