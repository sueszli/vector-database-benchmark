import numpy as np
from math import cos, sin, floor, sqrt, pi, ceil

def generate(shape, radius, k=32, seed=None):
    if False:
        while True:
            i = 10
    '\n    Generate blue noise over a two-dimensional rectangle of size (width,height)\n\n    Parameters\n    ----------\n\n    shape : tuple\n        Two-dimensional domain (width x height) \n    radius : float\n        Minimum distance between samples\n    k : int, optional\n        Limit of samples to choose before rejection (typically k = 30)\n    seed : int, optional\n        If provided, this will set the random seed before generating noise,\n        for valid pseudo-random comparisons.\n\n    References\n    ----------\n\n    .. [1] Fast Poisson Disk Sampling in Arbitrary Dimensions, Robert Bridson,\n           Siggraph, 2007. :DOI:`10.1145/1278780.1278807`\n    '

    def sqdist(a, b):
        if False:
            while True:
                i = 10
        ' Squared Euclidean distance '
        (dx, dy) = (a[0] - b[0], a[1] - b[1])
        return dx * dx + dy * dy

    def grid_coords(p):
        if False:
            return 10
        ' Return index of cell grid corresponding to p '
        return (int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize)))

    def fits(p, radius):
        if False:
            while True:
                i = 10
        ' Check whether p can be added to the queue '
        radius2 = radius * radius
        (gx, gy) = grid_coords(p)
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in range(max(gy - 2, 0), min(gy + 3, grid_height)):
                g = grid[x + y * grid_width]
                if g is None:
                    continue
                if sqdist(p, g) <= radius2:
                    return False
        return True
    if seed is not None:
        from numpy.random.mtrand import RandomState
        rng = RandomState(seed=seed)
    else:
        rng = np.random
    (width, height) = shape
    cellsize = radius / sqrt(2)
    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)
    p = rng.uniform(0, shape, 2)
    queue = [p]
    (grid_x, grid_y) = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p
    while queue:
        qi = rng.randint(len(queue))
        (qx, qy) = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            theta = rng.uniform(0, 2 * pi)
            r = radius * np.sqrt(rng.uniform(1, 4))
            p = (qx + r * cos(theta), qy + r * sin(theta))
            if not (0 <= p[0] < width and 0 <= p[1] < height) or not fits(p, radius):
                continue
            queue.append(p)
            (gx, gy) = grid_coords(p)
            grid[gx + gy * grid_width] = p
    return np.array([p for p in grid if p is not None])
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(1)
    fig = plt.figure(figsize=(9, 3.25))
    ax = plt.subplot(1, 3, 1, aspect=1, xlim=[0, 5], xticks=[], ylim=[0, 5], yticks=[])
    V = np.random.uniform(0, 5, (1600, 2))
    ax.scatter(V[:, 0], V[:, 1], s=5, edgecolor='None', facecolor='black')
    ax.set_title('Uniform distribution (n=1600)')
    ax = plt.subplot(1, 3, 2, aspect=1, xlim=[0, 5], xticks=[], ylim=[0, 5], yticks=[])
    (X, Y) = np.meshgrid(np.linspace(0, 5, 40), np.linspace(0, 5, 40))
    X += np.random.normal(0, 0.04, X.shape)
    Y += np.random.normal(0, 0.04, Y.shape)
    ax.scatter(X, Y, s=5, edgecolor='None', facecolor='black')
    ax.set_title('Jittered (n=1600)')
    ax = plt.subplot(1, 3, 3, aspect=1, xlim=[0, 5], xticks=[], ylim=[0, 5], yticks=[])
    V = generate([5, 5], 0.099)
    ax.scatter(V[:, 0], V[:, 1], s=5, edgecolor='None', facecolor='black')
    ax.set_title('Blue noise distribution (n=%d)' % len(V))
    plt.tight_layout()
    plt.savefig('../../figures/beyond/bluenoise.pdf')
    plt.show()