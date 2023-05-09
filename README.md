# 2AMC15-2023-DIC

Welcome to 2AMC15 Data Intelligence Challenge!.
This is the repository containing the challenge environment code.

## Quickstart

1. Create an agent which inherits from the `BaseAgent` class
2. Add the agents you want to test to `train.py`
   - There are 2 places to add you agent. Look for the comment `# Add your agent here` for where to add your agent.
3. Run `$ python train.py grid_configs/rooms-1.grd --out results/` to start training!

`train.py` is just an example training script. 
Feel free to modify it as necessary.
In our basic example, we use command line arguments to select options for it.
This may not be convenient for you and you can choose to replace this training script with whatever you want.
By default, its usage is:

```bash
usage: train.py [-h] [--no_gui] [--sigma SIGMA] [--fps FPS] [--iter ITER]
                [--random_seed RANDOM_SEED] [--out OUT]
                GRID [GRID ...]

DIC Reinforcement Learning Trainer.

positional arguments:
  GRID                  Paths to the grid file to use. There can be more than
                        one.

options:
  -h, --help            show this help message and exit
  --no_gui              Disables rendering to train faster
  --sigma SIGMA         Sigma value for the stochasticity of the environment.
  --fps FPS             Frames per second to render at. Only used if no_gui is
                        not set.
  --iter ITER           Number of iterations to go through.
  --random_seed RANDOM_SEED
                        Random seed value for the environment.
  --out OUT             Where to save training results.
```
## Code guide

The code is made up of 3 modules: 

1. `agent`
2. `level_editor`
3. `world`

### The `agent` module

The `agent` module contains the `BaseAgent` class as well as some benchmark agents to test against.

The `BaseAgent` is an abstract class and all RL agents for DIC must inherit from/implement it.
If you know/understand class inheritence, skip the following section

#### `BaseAgent` as an abstract class
Think of this like how all models in PyTorch start like 

```python
class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
    ...
```

In this case, `NewModel` inherits from `nn.Module`, which gives it the ability to do back propagation, store parameters, etc. without you having to manually code that every time.
It also ensures that every class that inherits from `nn.Module` contains _at least_ the `forward()` method, which allows a forward pass to actually happen.

In the case of your RL agent, inheriting from `BaseAgent` guarantees that your agent implements `process_reward()` and `take_action()`.
This ensures that no matter what RL agent you make and however you code it, the environment and training code can always interact with it in the same way.
Check out the benchmark agents to see examples.

### The `level_editor` module

The `level_editor` module contains a file called `app.py`.
Run this file to make new levels.

```bash
$ python app.py
```

This will start up a web server where you can edit levels.
To view the level editor itself, go to `127.0.0.1:5000`.
All levels will be saved to the `grid_configs/` directory.

Where the grids are saved can be changed in the file `level_editor/__init__.py`, but this is not recommended.

We also provide a `grid_generator.py` file to generate random grids, found in `level_editor` directory.
Usage is:

```bash
$ cd level_editor
$ python grid_generator.py 

usage: grid_generator.py [-h] N_GRIDS N_ROOMS FILE_PREFIX

Randomly generate grids.

positional arguments:
  N_GRIDS      Number of grids to generate.
  N_ROOMS      Number of rooms to generate in each grid.
  FILE_PREFIX  Prefix to give to the generated file name.

options:
  -h, --help   show this help message and exit
```

### The `world` module

The world module contains:
1. `environment.py`
2. `grid.py`
3. `gui.py`

#### The Environment

The environment is very important because it contains everything we hold dear, including ourselves [^1].
It is also the name of the class which our RL agent will act within.

The main interaction with `Environment` is through the methods:

- `Environment()` to initialize the environment
- `get_observation()` to get an environment observation without taking a step or resetting the environment.
- `reset()` to reset the environment
- `step()` to actually take a time step with the environment.

Explanations for each of these methods and how to use them can be found in the examples in the `environment.py` files and in the documentation in the code itself.

[^1]: In case you missed it, this sentence is a joke. Please do not write all your code in the `Environment` class.

#### The Grid

The `Grid` class is the world on which the agents actually move.
It is essentially a fancy Numpy array with different methods to make things easier for us to work with.

#### The GUI

The Graphical User Interface provides a way for you to actually see what the RL agent is doing.
While performant and written using PyGame, it is still about 1300x slower than not running a GUI.
Because of this, we recommend using it only while testing/debugging and not while training.

## Requirements

- python ~= 3.10
- numpy >= 1.24
- tqdm ~= 4
- pygame ~= 2.3
- flask ~= 2.2
- flask-socketio ~= 5.3
- pillow ~= 9.4
- colorcet ~=3.0

