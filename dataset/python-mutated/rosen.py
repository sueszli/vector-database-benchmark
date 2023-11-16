import numpy as np

def rosen(x):
    if False:
        while True:
            i = 10
    t0 = 100 * (x[1:] - x[:-1] ** 2) ** 2
    t1 = (1 - x[:-1]) ** 2
    return np.sum(t0 + t1)