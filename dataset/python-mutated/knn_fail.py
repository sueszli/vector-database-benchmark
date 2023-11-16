from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt
from knn import KNN

def get_data():
    if False:
        i = 10
        return i + 15
    width = 8
    height = 8
    N = width * height
    X = np.zeros((N, 2))
    Y = np.zeros(N)
    n = 0
    start_t = 0
    for i in range(width):
        t = start_t
        for j in range(height):
            X[n] = [i, j]
            Y[n] = t
            n += 1
            t = (t + 1) % 2
        start_t = (start_t + 1) % 2
    return (X, Y)
if __name__ == '__main__':
    (X, Y) = get_data()
    plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
    plt.show()
    model = KNN(3)
    model.fit(X, Y)
    print('Train accuracy:', model.score(X, Y))