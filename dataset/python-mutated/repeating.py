import numpy as np

def repeating(x, nvar_y):
    if False:
        for i in range(10):
            print('nop')
    nvar_x = x.shape[0]
    y = np.empty(nvar_x * (1 + nvar_y))
    y[0:nvar_x] = x[0:nvar_x]
    y[nvar_x:] = np.repeat(x, nvar_y)
    return y