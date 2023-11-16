from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
N = 100
D = 2
N_per_class = N // 2
X = np.random.randn(N, D)
X[:N_per_class, :] = X[:N_per_class, :] - 2 * np.ones((N_per_class, D))
X[N_per_class:, :] = X[N_per_class:, :] + 2 * np.ones((N_per_class, D))
T = np.array([0] * N_per_class + [1] * N_per_class)
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)
w = np.random.randn(D + 1)
z = Xb.dot(w)

def sigmoid(z):
    if False:
        print('Hello World!')
    return 1 / (1 + np.exp(-z))
Y = sigmoid(z)

def cross_entropy(T, Y):
    if False:
        print('Hello World!')
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E
learning_rate = 0.1
for i in range(100):
    if i % 10 == 0:
        print(cross_entropy(T, Y))
    w += learning_rate * Xb.T.dot(T - Y)
    Y = sigmoid(Xb.dot(w))
print('Final w:', w)
plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 100)
y_axis = -(w[0] + x_axis * w[1]) / w[2]
plt.plot(x_axis, y_axis)
plt.show()