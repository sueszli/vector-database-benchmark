import numpy as np

def create_grid(x):
    if False:
        return 10
    N = x.shape[0]
    z = np.zeros((N, N, 3))
    z[:, :, 0] = x.reshape(-1, 1)
    z[:, :, 1] = x
    fast_grid = z.reshape(N * N, 3)
    return fast_grid