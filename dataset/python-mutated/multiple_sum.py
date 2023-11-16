import numpy as np

def multiple_sum(array):
    if False:
        return 10
    rows = array.shape[0]
    cols = array.shape[1]
    out = np.zeros((rows, cols))
    for row in range(0, rows):
        out[row, :] = np.sum(array - array[row, :], 0)
    return out