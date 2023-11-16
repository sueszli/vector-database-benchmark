from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
N = 4
D = 2
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([0, 1, 1, 0])
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)
w = np.random.randn(D + 1)
z = Xb.dot(w)

def sigmoid(z):
    if False:
        while True:
            i = 10
    return 1 / (1 + np.exp(-z))
Y = sigmoid(z)

def cross_entropy(T, Y):
    if False:
        return 10
    return -(T * np.log(Y) + (1 - T) * np.log(1 - Y)).sum()
learning_rate = 0.001
error = []
w_mags = []
for i in range(100000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 1000 == 0:
        print(e)
    w += learning_rate * Xb.T.dot(T - Y)
    w_mags.append(w.dot(w))
    Y = sigmoid(Xb.dot(w))
plt.plot(error)
plt.title('Cross-entropy per iteration')
plt.show()
plt.plot(w_mags)
plt.title('w^2 magnitudes')
plt.show()
print('Final w:', w)
print('Final classification rate:', 1 - np.abs(T - np.round(Y)).sum() / N)