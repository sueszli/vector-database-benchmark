import numpy as np

def laplacian(grid):
    if False:
        return 10
    return np.roll(grid, +1, 0) + np.roll(grid, -1, 0) + np.roll(grid, +1, 1) + np.roll(grid, -1, 1) - 4 * grid

def evolve(grid, dt, D=1):
    if False:
        i = 10
        return i + 15
    return grid + dt * D * laplacian(grid)