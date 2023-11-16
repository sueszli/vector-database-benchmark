from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt

def get_spiral():
    if False:
        for i in range(10):
            print('nop')
    radius = np.linspace(1, 10, 100)
    thetas = np.empty((6, 100))
    for i in range(6):
        start_angle = np.pi * i / 3.0
        end_angle = start_angle + np.pi / 2
        points = np.linspace(start_angle, end_angle, 100)
        thetas[i] = points
    x1 = np.empty((6, 100))
    x2 = np.empty((6, 100))
    for i in range(6):
        x1[i] = radius * np.cos(thetas[i])
        x2[i] = radius * np.sin(thetas[i])
    X = np.empty((600, 2))
    X[:, 0] = x1.flatten()
    X[:, 1] = x2.flatten()
    X += np.random.randn(600, 2) * 0.5
    Y = np.array([0] * 100 + [1] * 100 + [0] * 100 + [1] * 100 + [0] * 100 + [1] * 100)
    return (X, Y)
(X, Y) = get_spiral()
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()