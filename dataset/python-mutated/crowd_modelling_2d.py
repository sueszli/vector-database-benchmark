"""Mean Field Crowd Modelling Game in 2d.

Please see the C++ implementation under games/mfg/crowd_modelling_2d.h for
more information.
"""
from typing import Sequence

def grid_to_forbidden_states(grid: Sequence[str]) -> str:
    if False:
        return 10
    "Converts a grid into string representation of forbidden states.\n\n  Args:\n    grid: Rows of the grid. '#' character denotes a forbidden state. All rows\n      should have the same number of columns, i.e. cells.\n\n  Returns:\n    String representation of forbidden states in the form of x (column) and y\n    (row) pairs, e.g. [1|1;0|2].\n  "
    forbidden_states = []
    num_cols = len(grid[0])
    for (y, row) in enumerate(grid):
        assert len(row) == num_cols, f'Number of columns should be {num_cols}.'
        for (x, cell) in enumerate(row):
            if cell == '#':
                forbidden_states.append(f'{x}|{y}')
    return '[' + ';'.join(forbidden_states) + ']'
FOUR_ROOMS_FORBIDDEN_STATES = grid_to_forbidden_states(['#############', '#     #     #', '#     #     #', '#           #', '#     #     #', '#     #     #', '### ##### ###', '#     #     #', '#     #     #', '#           #', '#     #     #', '#     #     #', '#############'])
FOUR_ROOMS = {'forbidden_states': FOUR_ROOMS_FORBIDDEN_STATES, 'horizon': 40, 'initial_distribution': '[1|1]', 'initial_distribution_value': '[1.0]', 'size': 13}
MAZE_FORBIDDEN_STATES = grid_to_forbidden_states(['######################', '#      #     #     # #', '#      #     #     # #', '######    #  # ##  # #', '#         #  # #   # #', '#         #  # ### # #', '#  ########  #   #   #', '#    # # #  ##   #   #', '#    # # #     # # ###', '#    # # #     # # # #', '###### # ####### # # #', '#  #         #   # # #', '#  # ## ###  #   # # #', '## # #    #  ##### # #', '## # # #  #      # # #', '#    # ####        # #', '# ####  # ########   #', '#       #  #   # ### #', '#  #  # #  # # #   # #', '# ##### #    # #     #', '#            #       #', '######################'])
MAZE = {'forbidden_states': MAZE_FORBIDDEN_STATES, 'horizon': 100, 'initial_distribution': '[1|1]', 'initial_distribution_value': '[1.0]', 'size': 22}