"""Path Visualizer.

This script is used to visualize the path of the agents in the environment.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from colorcet import bmw
from PIL import Image
from PIL import ImageDraw
import numpy as np

from world import Grid
from world import EnvironmentGUI


def draw_base_image(cells: np.ndarray,
                    scalar: int,
                    image_size: tuple[int, int]) -> Image.Image:
    """Draws the base image containing the grid and objects on the grid.

    Args:
        cells: The cell array underlying the grid representation of the
            environment.
        scalar: How much to scale the original grid by in the output image.
        image_size: Output image size.

    Returns:
        An RGBA image with the grid on it.
    """
    grid_size = cells.shape
    base_image = Image.new(mode="RGBA", size=image_size,
                           color=(255, 255, 255, 255))
    draw = ImageDraw.ImageDraw(base_image)
    for row in range(grid_size[1]):
        y = (row * scalar) + 1
        for col in range(grid_size[0]):
            x = (col * scalar) + 1
            val = cells[col, row]
            color = EnvironmentGUI.CELL_COLORS[val]

            draw.rectangle((x, y, x + scalar, y + scalar),
                           color,
                           outline=(255, 255, 255))

    return base_image


def draw_freq_image(agent_path: list[tuple[int, int]],
                    grid_shape: tuple[int, int],
                    grid_scalar: int,
                    freq_scalar: int,
                    image_size: tuple[int, int]) -> Image.Image:
    """Draws the cell visit frequency image."""
    # Create the frequency grid array to figure out how often a cell if
    # traversed
    freq_grid = np.zeros(grid_shape, dtype=float)
    for pos in agent_path:
        freq_grid[pos] += 1.

    # Normalize by the max value to 0-255
    freq_grid /= np.max(freq_grid)
    freq_grid *= 255.
    freq_grid = freq_grid.astype(int)

    cell_offset = (grid_scalar - freq_scalar) // 2

    freq_image = Image.new(mode="RGBA", size=image_size,
                           color=(255, 255, 255, 0))
    draw = ImageDraw.ImageDraw(freq_image)

    for row in range(grid_shape[1]):
        y = (row * grid_scalar) + 1
        for col in range(grid_shape[0]):
            x = (col * grid_scalar) + 1
            val = freq_grid[col, row]
            if val == 0:
                # Don't draw anything if the cells has never been traversed.
                continue
            try:
                # -1 here because we consider 0 non-traversed and is not drawn.
                color = bmw[val - 1]
            except IndexError:
                # There is no chance the value is < 1 here, but just in case.
                color = bmw[0]

            draw.rectangle((x + cell_offset, y + cell_offset,
                            x + freq_scalar, y + freq_scalar),
                           color)

    return freq_image


def visualize_path(grid_cells: np.ndarray,
                   agent_paths: list[list[tuple[int, int]]]) \
        -> list[Image.Image]:
    """Visualizes the path of (multiple) agents through the environment.

    Args:
        grid_cells: The grid cells that are underlying the Grid object.
        agent_paths: A list of tuples containing the x and y coordinates of
            the agent's path.

    Returns:
        A list of images showing the grid and the frequency of the agent
        traversing each position on the grid.
    """
    grid_size = grid_cells.shape
    scalar = 30
    image_size = tuple((g * scalar) + 2 for g in grid_size)
    freq_scalar = 20
    final_images = []

    base_image = draw_base_image(grid_cells, scalar, image_size)

    for i, agent_path in enumerate(agent_paths):
        freq_image = draw_freq_image(agent_path, grid_size, scalar, freq_scalar,
                                     image_size)
        final_images.append(Image.alpha_composite(base_image, freq_image))
    return final_images
