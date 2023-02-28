from pathlib import Path


GRID_CONFIGS_FP = Path.cwd().parent / Path("grid_configs")
GRID_CONFIGS_FP.mkdir(parents=True, exist_ok=True)

__all__ = ["GRID_CONFIGS_FP"]
