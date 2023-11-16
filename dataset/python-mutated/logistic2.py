from __future__ import print_function, division
from builtins import range
import numpy as np
N = 100
D = 2
X = np.random.randn(N, D)
X[:50, :] = X[:50, :] - 2 * np.ones((50, D))
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))
T = np.array([0] * 50 + [1] * 50)
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)
w = np.random.randn(D + 1)
z = Xb.dot(w)

def sigmoid(z):
    if False:
        return 10
    return 1 / (1 + np.exp(-z))
Y = sigmoid(z)

def cross_entropy(T, Y):
    if False:
        while True:
            i = 10
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E
print(cross_entropy(T, Y))
w = np.array([0, 4, 4])
z = Xb.dot(w)
Y = sigmoid(z)
print(cross_entropy(T, Y))