from __future__ import print_function, division
from builtins import range
import numpy as np
N = 100
D = 2
X = np.random.randn(N, D)
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)
w = np.random.randn(D + 1)
z = Xb.dot(w)

def sigmoid(z):
    if False:
        return 10
    return 1 / (1 + np.exp(-z))
print(sigmoid(z))