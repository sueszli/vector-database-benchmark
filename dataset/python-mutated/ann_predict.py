from __future__ import print_function, division
from builtins import range
import numpy as np
from process import get_data
(X, Y, _, _) = get_data()
M = 5
D = X.shape[1]
K = len(set(Y))
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

def softmax(a):
    if False:
        for i in range(10):
            print('nop')
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    if False:
        i = 10
        return i + 15
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)
P_Y_given_X = forward(X, W1, b1, W2, b2)
print('P_Y_given_X.shape:', P_Y_given_X.shape)
predictions = np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y, P):
    if False:
        while True:
            i = 10
    return np.mean(Y == P)
print('Score:', classification_rate(Y, predictions))