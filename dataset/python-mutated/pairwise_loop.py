import numpy as np

def pairwise_loop(X):
    if False:
        for i in range(10):
            print('nop')
    (M, N) = X.shape
    D = np.empty((M, M))
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D