"""Grid Generator.

This is an example of a programmatic approach to grid generation. You can use
this as inspiration for randomly generated grids. Credit to Tom v. Meer for
writing this.

This script generates 5 grids, each with 5 rooms.
"""
import numpy as np

from world import Grid
from level_editor import GRID_CONFIGS_FP


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
    # Create the corridor:
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

    # Save the grid
    grid.save_grid_file(GRID_CONFIGS_FP / f"{grid_name}.grd")


if __name__ == '__main__':
    for grid_num in range(5):
        generate_random_grid(5, f"random-house-{grid_num}")
