import numpy as np

def allpairs_distances_loops(X, Y):
    if False:
        print('Hello World!')
    result = np.zeros((X.shape[0], Y.shape[0]), X.dtype)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            result[i, j] = np.sum((X[i, :] - Y[j, :]) ** 2)
    return result