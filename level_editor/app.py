"""Level Editor.

Level editor so making grids is somewhat easier than writing it as a raw numpy
array. Credit to Tom v. Meer for writing this up.
"""
import pickle
import ast

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

from dic import Grid
from level_editor import GRID_CONFIGS_FP


# Initialize SocketIO App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socket_io = SocketIO(app)


def draw_grid(grid):
    """Creates a JSON payload which will be displayed in the browser."""
    materials = {0: 'cell_empty',
                 1: 'cell_wall',
                 2: 'cell_obstacle',
                 3: 'cell_dirt',
                 4: 'cell_charger'}
    return {'grid': render_template('grid.html',
                                    height=30, width=30,
                                    n_rows=grid.n_rows, n_cols=grid.n_cols,
                                    room_config=grid.cells,
                                    materials=materials)}


# Routes
@app.route('/')
def home():
    return render_template("editor.html")


@app.route('/build_grid')
def build_grid():
    """Main route for building a grid.

    Given a request with the following parameters, a grid and accompanying
    statistics are being constructed.

    Request params:
        height: number of rows in the grid.
        width: number of columns in the grid.
        obstacles: a list of tuples (x,y) of obstacle locations.
        goals: a list of tuples (x,y) of goal locations.
        deaths: a list of tuples (x,y) of death-tile locations.
        save: boolean (true, false) to save the current grid to a file.
        name: filename to save the current grid to.
     """
    n_rows = int(request.args.get('height'))
    n_cols = int(request.args.get('width'))
    obstacles = ast.literal_eval(request.args.get('obstacles'))
    goals = ast.literal_eval(request.args.get('goals'))
    chargers = ast.literal_eval(request.args.get('chargers'))
    to_save = False if request.args.get('save') == 'false' else True
    name = str(request.args.get('name'))
    grid = Grid(n_cols, n_rows)
    for (x, y) in obstacles:
        grid.place_single_obstacle(x, y)
    for (x, y) in goals:
        grid.place_single_dirt(x, y)
    for (x, y) in chargers:
        grid.place_single_charger(x, y)

    drawn_grid = draw_grid(grid)
    if to_save and len(name) > 0:
        save_fp = GRID_CONFIGS_FP / f"{name}.grid"
        with open(save_fp, "wb") as f:
            pickle.dump(grid, f)
        drawn_grid["success"] = "true"
        drawn_grid["save_fp"] = str(save_fp)

    return drawn_grid


if __name__ == '__main__':
    socket_io.run(app, debug=True, allow_unsafe_werkzeug=True)
