"""Grid.

Grid specifically for the level editor. Credit to Tom v. Meer for writing this.
"""
import pickle
from pathlib import Path

import numpy as np


class Grid:
    def __init__(self, n_cols, n_rows):
        """Grid representation of the world.

        Possible grid values are:
        - 0: Empty
        - 1: Wall
        - 2: Obstacle
        - 3: Dirt
        - 4: Charger

        Args:
            n_cols: Number of columns the grid should have.
            n_rows: Number of rows the grid should have.
        """
        self.n_rows = n_rows
        self.n_cols = n_cols

        # Building the boundary of the grid:
        self.cells = np.zeros((n_cols, n_rows), dtype=np.int8)
        self.cells[0, :] = self.cells[-1, :] = 1
        self.cells[:, 0] = self.cells[:, -1] = 1

    def place_obstacle(self, x0, x1, y0, y1, from_edge=1):
        """Places a larger obstacle.

        Args:
            x0: Starting x coords of obstacle
            x1: Ending x coords of obstacle
            y0: Starting y coords of obstacle
            y1: Ending y coords of obstacle
            from_edge: Padding from the edge of the grid.
        """
        # Recalculate all coordinates with padding
        x0 = max(x0, from_edge)
        x1 = min(x1 + 1, self.n_cols - from_edge)
        y0 = max(y0, from_edge)
        y1 = min(y1 + 1, self.n_rows - from_edge)
        self.cells[x0:x1, y0:y1] = 2

    def place_single_obstacle(self, x, y):
        """Places a single obstacle marker on the grid."""
        self.cells[x][y] = 2

    def place_single_dirt(self, x, y):
        """Places a single dirt marker on the grid."""
        self.cells[x][y] = 3

    def place_single_charger(self, x, y):
        """Places a single charger marker on the grid."""
        self.cells[x][y] = 4

    def sum_dirt(self):
        """Counts the number of dirt particles in the grid."""
        return np.sum(self.cells == 3)

    def remove_dirt(self, x, y):
        """Removes dirt at x, y location on the grid."""
        self.cells[x][y] = 0


def load_grid_file(fp: Path) -> Grid:
    """Loads a `.grid` file."""
    with open(fp, "rb") as f:
        grid = pickle.load(f)
    return grid
