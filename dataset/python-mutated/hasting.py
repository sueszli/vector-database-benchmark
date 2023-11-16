import numpy as np

def hasting(y, t, a1, a2, b1, b2, d1, d2):
    if False:
        for i in range(10):
            print('nop')
    yprime = np.empty((3,))
    yprime[0] = y[0] * (1.0 - y[0]) - a1 * y[0] * y[1] / (1.0 + b1 * y[0])
    yprime[1] = a1 * y[0] * y[1] / (1.0 + b1 * y[0]) - a2 * y[1] * y[2] / (1.0 + b2 * y[1]) - d1 * y[1]
    yprime[2] = a2 * y[1] * y[2] / (1.0 + b2 * y[1]) - d2 * y[2]
    return yprime