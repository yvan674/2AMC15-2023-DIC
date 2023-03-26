"""Grid Generator.

This is an example of a programmatic approach to grid generation. You can use
this as inspiration for randomly generated grids. Credit to Tom v. Meer for
writing this.

This script generates 5 grids, each with 5 rooms.
"""
from argparse import ArgumentParser
from random import randint

from tqdm import trange
import numpy as np

from world import Grid
from level_editor import GRID_CONFIGS_FP


def parse_args():
    p = ArgumentParser(description="Randomly generate grids.")

    p.add_argument("N_GRIDS", type=int,
                   help="Number of grids to generate.")
    p.add_argument("N_ROOMS", type=int,
                   help="Number of rooms to generate in each grid.")
    p.add_argument("FILE_PREFIX", type=str,
                   help="Prefix to give to the generated file name.")

    return p.parse_args()


def generate_random_grid(n_rooms: int, grid_name: str):
    """Generates a random grid with n rooms.

    Args:
        n_rooms: Number of rooms to generate on the grid.
        grid_name: Name of the grid for saving. This is without the `.grid`
            filetype suffix
    """
    # Get a grid with a random height and width:
    height = np.random.randint(10, 20)
    width = np.random.randint(10, 20)
    grid = Grid(width, height)

    # Create the corridor
    corr_y0 = int(height / 2)
    corr_y1 = int(height / 2) + 2

    # Generate upper rooms
    rooms = np.random.randint(2, n_rooms)
    for i in range(0, width, width//rooms):
        grid.place_obstacle(x0=i,
                            x1=i + (width // rooms) - 2,
                            y0=corr_y0,
                            y1=corr_y0)
    for i in range(0, rooms-1):
        grid.place_obstacle(x0=(i + 1) * int(width / rooms),
                            x1=(i + 1) * int(width / rooms),
                            y0=1,
                            y1=corr_y0)

    # Generate lower rooms
    rooms = np.random.randint(2, n_rooms)
    for i in range(0, width, width // rooms):
        grid.place_obstacle(x0=i,
                            x1=i + (width // rooms) - 2,
                            y0=corr_y1,
                            y1=corr_y1)
    for i in range(0, rooms - 1):
        grid.place_obstacle(x0=(i + 1) * int(width / rooms),
                            x1=(i + 1) * int(width / rooms),
                            y0=corr_y1,
                            y1=height)

    def choose_empty_cell():
        zeros = np.where(grid.cells == 0)
        idx = randint(0, len(zeros[0]) - 1)
        return zeros[0][idx], zeros[1][idx]

    # Place dirt
    num_dirt = (height * width) // 10

    for _ in range(num_dirt):
        dirt_x, dirt_y = choose_empty_cell()
        grid.place_single_dirt(dirt_x, dirt_y)

    # Place charger
    charger_x, charger_y = choose_empty_cell()
    grid.place_single_charger(charger_x, charger_y)

    # Save the grid
    grid.save_grid_file(GRID_CONFIGS_FP / f"{grid_name}.grd")


if __name__ == '__main__':
    args = parse_args()
    for grid_num in trange(args.N_GRIDS):
        generate_random_grid(args.N_ROOMS, f"{args.FILE_PREFIX}-{grid_num}")
