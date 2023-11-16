from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
N = 4
D = 2
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([0, 1, 1, 0])
ones = np.ones((N, 1))
xy = (X[:, 0] * X[:, 1]).reshape(N, 1)
Xb = np.concatenate((ones, xy, X), axis=1)
w = np.random.randn(D + 2)
z = Xb.dot(w)

def sigmoid(z):
    if False:
        return 10
    return 1 / (1 + np.exp(-z))
Y = sigmoid(z)

def cross_entropy(T, Y):
    if False:
        for i in range(10):
            print('nop')
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E
learning_rate = 0.01
error = []
for i in range(10000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 1000 == 0:
        print(e)
    w += learning_rate * (Xb.T.dot(T - Y) - 0.01 * w)
    Y = sigmoid(Xb.dot(w))
plt.plot(error)
plt.title('Cross-entropy per iteration')
plt.show()
print('Final w:', w)
print('Final classification rate:', 1 - np.abs(T - np.round(Y)).sum() / N)