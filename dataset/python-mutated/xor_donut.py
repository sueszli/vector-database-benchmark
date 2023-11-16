from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1, W2, b2):
    if False:
        while True:
            i = 10
    Z = X.dot(W1) + b1
    Z = Z * (Z > 0)
    activation = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-activation))
    return (Y, Z)

def predict(X, W1, b1, W2, b2):
    if False:
        i = 10
        return i + 15
    (Y, _) = forward(X, W1, b1, W2, b2)
    return np.round(Y)

def derivative_w2(Z, T, Y):
    if False:
        return 10
    return (T - Y).dot(Z)

def derivative_b2(T, Y):
    if False:
        for i in range(10):
            print('nop')
    return (T - Y).sum()

def derivative_w1(X, Z, T, Y, W2):
    if False:
        i = 10
        return i + 15
    dZ = np.outer(T - Y, W2) * (Z > 0)
    return X.T.dot(dZ)

def derivative_b1(Z, T, Y, W2):
    if False:
        print('Hello World!')
    dZ = np.outer(T - Y, W2) * (Z > 0)
    return dZ.sum(axis=0)

def get_log_likelihood(T, Y):
    if False:
        return 10
    return np.sum(T * np.log(Y) + (1 - T) * np.log(1 - Y))

def test_xor():
    if False:
        for i in range(10):
            print('nop')
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    W1 = np.random.randn(2, 5)
    b1 = np.zeros(5)
    W2 = np.random.randn(5)
    b2 = 0
    LL = []
    learning_rate = 0.01
    regularization = 0.0
    last_error_rate = None
    for i in range(30000):
        (pY, Z) = forward(X, W1, b1, W2, b2)
        ll = get_log_likelihood(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.mean(prediction != Y)
        LL.append(ll)
        gW2 = derivative_w2(Z, Y, pY)
        gb2 = derivative_b2(Y, pY)
        gW1 = derivative_w1(X, Z, Y, pY, W2)
        gb1 = derivative_b1(Z, Y, pY, W2)
        W2 += learning_rate * (gW2 - regularization * W2)
        b2 += learning_rate * (gb2 - regularization * b2)
        W1 += learning_rate * (gW1 - regularization * W1)
        b1 += learning_rate * (gb1 - regularization * b1)
        if i % 1000 == 0:
            print(ll)
    print('final classification rate:', np.mean(prediction == Y))
    plt.plot(LL)
    plt.show()

def test_donut():
    if False:
        print('Hello World!')
    N = 1000
    R_inner = 5
    R_outer = 10
    R1 = np.random.randn(N // 2) + R_inner
    theta = 2 * np.pi * np.random.random(N // 2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
    R2 = np.random.randn(N // 2) + R_outer
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T
    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0] * (N // 2) + [1] * (N // 2))
    n_hidden = 8
    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)
    LL = []
    learning_rate = 5e-05
    regularization = 0.2
    last_error_rate = None
    for i in range(3000):
        (pY, Z) = forward(X, W1, b1, W2, b2)
        ll = get_log_likelihood(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()
        LL.append(ll)
        gW2 = derivative_w2(Z, Y, pY)
        gb2 = derivative_b2(Y, pY)
        gW1 = derivative_w1(X, Z, Y, pY, W2)
        gb1 = derivative_b1(Z, Y, pY, W2)
        W2 += learning_rate * (gW2 - regularization * W2)
        b2 += learning_rate * (gb2 - regularization * b2)
        W1 += learning_rate * (gW1 - regularization * W1)
        b1 += learning_rate * (gb1 - regularization * b1)
        if i % 300 == 0:
            print('i:', i, 'll:', ll, 'classification rate:', 1 - er)
    plt.plot(LL)
    plt.show()
if __name__ == '__main__':
    test_xor()