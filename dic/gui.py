"""GUI.

Provides a GUI for the environment using pygame.
"""
import pygame
import sys


class EnvironmentGUI:
    def __init__(self, grid_size: tuple[int, int],
                 window_size: tuple[int, int] = (1152, 768)):
        """Provides a GUI to show what is happening in the environment.

        Args:
            grid_size: (n_cols, n_rows) in the grid.
            window_size: The size of the pygame window.
        """
        self.grid_size = grid_size

        # Calculate grid size
        if max(grid_size) <= 20:
            self.cell_size = 30
        elif 20 < max(grid_size) <= 30:
            self.cell_size = 20
        elif 30 < max(grid_size) <= 45:
            self.cell_size = 15
        elif 45 < max(grid_size) <= 55:
            self.cell_size = 10
        elif 55 < max(grid_size) <= 120:
            self.cell_size = 5
        else:
            raise ValueError(f"Grid size of {grid_size} is too big to "
                             f"visualize with our code.")


        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Data Intelligence Challenge 2023")
        self.clock = pygame.time.Clock()

        self.stats = self._reset_stats()

        self._initial_render()

    def reset(self):
        """Called during the reset method of the environment."""
        self.stats = self._reset_stats()
        self._initial_render()

    @staticmethod
    def _reset_stats():
        return {"total_dirt_collected": 0,
                "total_failed_move": 0,
                "total_done": 0,
                "fps": "0.0"}

    def _initial_render(self):
        """Initial render of the environment. Also shows loading text."""
        background = pygame.Surface(self.window.get_size())
        background = background.convert()
        background.fill((250, 250, 250))

        # Display the loading text
        font = pygame.font.Font(None, 36)
        text = font.render("Loading environment...", True, (10, 10, 10))
        textpos = text.get_rect()
        textpos.centerx = background.get_rect().centerx
        textpos.centery = background.get_rect().centery

        # Blit the text onto the background surface
        background.blit(text, textpos)
        # Blit the background onto the window
        update_rect = self.window.blit(background, background.get_rect())
        # Tell pygame to update the display where the window was blit-ed
        pygame.display.update(update_rect)

    @staticmethod
    def _downsample_rect(rect: pygame.Rect, scalar: float) -> pygame.Rect:
        """Downsamples the given rectangle by a scalar."""
        x = rect.x * scalar
        y = rect.y * scalar
        width = rect.width * scalar
        height = rect.height * scalar
        return pygame.Rect(x, y, width, height)

    def _draw_grid(self, surface: pygame.Surface, scalar: float):
        """Draws the grid world on the given surface."""
        pass

    def _draw_agent(self, surface: pygame.Surface, scalar: float):
        """Draws the agent on the grid world."""
        pass

    def _draw_info(self, surface, x_pos, y_pos):
        """Draws the info panel on the surface."""
        pass

    def render(self):
        self.stats["fps"] = f"{self.clock.get_fps():.1f}"

        # Find the smallest window dimension and max grid size to calculate the
        # grid scalar
        scalar = min(self.window.get_size())
        scalar /= max(self.grid_size) * self.cell_size

        # Create a surface to actually draw on
        background = pygame.Surface(self.window.get_size()).convert()
        background.fill((250, 250, 250))

        # Draw each layer on the surface
        self._draw_grid(background, scalar)
        self._draw_agent(background, scalar)
        self._draw_info(background, 798, 30)

        # Blit the surface onto the window
        update_rect = self.window.blit(background, background.get_rect())
        pygame.display.update(update_rect)

        # Parse events that happened since the last render step
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.event.pump()
