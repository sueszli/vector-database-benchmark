from __future__ import print_function, division
from builtins import range
import numpy as np
from process import get_binary_data
(X, Y, _, _) = get_binary_data()
D = X.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(a):
    if False:
        for i in range(10):
            print('nop')
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    if False:
        for i in range(10):
            print('nop')
    return sigmoid(X.dot(W) + b)
P_Y_given_X = forward(X, W, b)
predictions = np.round(P_Y_given_X)

def classification_rate(Y, P):
    if False:
        for i in range(10):
            print('nop')
    return np.mean(Y == P)
print('Score:', classification_rate(Y, predictions))