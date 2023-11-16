from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process import get_binary_data
(Xtrain, Ytrain, Xtest, Ytest) = get_binary_data()
D = Xtrain.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(a):
    if False:
        return 10
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    if False:
        i = 10
        return i + 15
    return sigmoid(X.dot(W) + b)

def classification_rate(Y, P):
    if False:
        print('Hello World!')
    return np.mean(Y == P)

def cross_entropy(T, pY):
    if False:
        while True:
            i = 10
    return -np.mean(T * np.log(pY) + (1 - T) * np.log(1 - pY))
train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)
    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)
    W -= learning_rate * Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate * (pYtrain - Ytrain).sum()
    if i % 1000 == 0:
        print(i, ctrain, ctest)
print('Final train classification_rate:', classification_rate(Ytrain, np.round(pYtrain)))
print('Final test classification_rate:', classification_rate(Ytest, np.round(pYtest)))
plt.plot(train_costs, label='train cost')
plt.plot(test_costs, label='test cost')
plt.legend()
plt.show()