import numpy as np

def periodic_dist(x, y, z, L, periodicX, periodicY, periodicZ):
    if False:
        print('Hello World!')
    'Computes distances between all particles and places the result\n    in a matrix such that the ij th matrix entry corresponds to the\n    distance between particle i and j'
    N = len(x)
    xtemp = np.tile(x, (N, 1))
    dx = xtemp - xtemp.T
    ytemp = np.tile(y, (N, 1))
    dy = ytemp - ytemp.T
    ztemp = np.tile(z, (N, 1))
    dz = ztemp - ztemp.T
    if periodicX:
        dx[dx > L / 2] = dx[dx > L / 2] - L
        dx[dx < -L / 2] = dx[dx < -L / 2] + L
    if periodicY:
        dy[dy > L / 2] = dy[dy > L / 2] - L
        dy[dy < -L / 2] = dy[dy < -L / 2] + L
    if periodicZ:
        dz[dz > L / 2] = dz[dz > L / 2] - L
        dz[dz < -L / 2] = dz[dz < -L / 2] + L
    d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    d[d == 0] = -1
    return (d, dx, dy, dz)