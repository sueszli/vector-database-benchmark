from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from util import get_spiral, get_xor, get_donut, get_clouds, plot_decision_boundary
import numpy as np
import matplotlib.pyplot as plt

def linear(X1, X2, c=0):
    if False:
        print('Hello World!')
    return X1.dot(X2.T) + c

def rbf(X1, X2, gamma=None):
    if False:
        while True:
            i = 10
    if gamma is None:
        gamma = 1.0 / X1.shape[-1]
    if np.ndim(X1) == 1 and np.ndim(X2) == 1:
        result = np.exp(-gamma * np.linalg.norm(X1 - X2) ** 2)
    elif np.ndim(X1) > 1 and np.ndim(X2) == 1 or (np.ndim(X1) == 1 and np.ndim(X2) > 1):
        result = np.exp(-gamma * np.linalg.norm(X1 - X2, axis=1) ** 2)
    elif np.ndim(X1) > 1 and np.ndim(X2) > 1:
        result = np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) ** 2)
    return result

def sigmoid(X1, X2, gamma=0.05, c=1):
    if False:
        print('Hello World!')
    return np.tanh(gamma * X1.dot(X2.T) + c)

class SVM:

    def __init__(self, kernel, C=1.0):
        if False:
            print('Hello World!')
        self.kernel = kernel
        self.C = C

    def _train_objective(self):
        if False:
            i = 10
            return i + 15
        return np.sum(self.alphas) - 0.5 * np.sum(self.YYK * np.outer(self.alphas, self.alphas))

    def fit(self, X, Y, lr=1e-05, n_iters=400):
        if False:
            i = 10
            return i + 15
        self.Xtrain = X
        self.Ytrain = Y
        self.N = X.shape[0]
        self.alphas = np.random.random(self.N)
        self.b = 0
        self.K = self.kernel(X, X)
        self.YY = np.outer(Y, Y)
        self.YYK = self.K * self.YY
        losses = []
        for _ in range(n_iters):
            loss = self._train_objective()
            losses.append(loss)
            grad = np.ones(self.N) - self.YYK.dot(self.alphas)
            self.alphas += lr * grad
            self.alphas[self.alphas < 0] = 0
            self.alphas[self.alphas > self.C] = self.C
        idx = np.where(self.alphas > 0 & (self.alphas < self.C))[0]
        bs = Y[idx] - (self.alphas * Y).dot(self.kernel(X, X[idx]))
        self.b = np.mean(bs)
        plt.plot(losses)
        plt.title('loss per iteration')
        plt.show()

    def _decision_function(self, X):
        if False:
            print('Hello World!')
        return (self.alphas * self.Ytrain).dot(self.kernel(self.Xtrain, X)) + self.b

    def predict(self, X):
        if False:
            return 10
        return np.sign(self._decision_function(X))

    def score(self, X, Y):
        if False:
            for i in range(10):
                print('nop')
        P = self.predict(X)
        return np.mean(Y == P)

def medical():
    if False:
        print('Hello World!')
    data = load_breast_cancer()
    (X, Y) = (data.data, data.target)
    (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=0.33)
    return (Xtrain, Xtest, Ytrain, Ytest, rbf, 0.001, 200)

def medical_sigmoid():
    if False:
        while True:
            i = 10
    data = load_breast_cancer()
    (X, Y) = (data.data, data.target)
    (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=0.33)
    return (Xtrain, Xtest, Ytrain, Ytest, sigmoid, 0.001, 200)

def xor():
    if False:
        print('Hello World!')
    (X, Y) = get_xor()
    (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=0.33)
    kernel = lambda X1, X2: rbf(X1, X2, gamma=5.0)
    return (Xtrain, Xtest, Ytrain, Ytest, kernel, 0.01, 300)

def donut():
    if False:
        i = 10
        return i + 15
    (X, Y) = get_donut()
    (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=0.33)
    kernel = lambda X1, X2: rbf(X1, X2, gamma=5.0)
    return (Xtrain, Xtest, Ytrain, Ytest, kernel, 0.01, 300)

def spiral():
    if False:
        for i in range(10):
            print('nop')
    (X, Y) = get_spiral()
    (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=0.33)
    kernel = lambda X1, X2: rbf(X1, X2, gamma=5.0)
    return (Xtrain, Xtest, Ytrain, Ytest, kernel, 0.01, 300)

def clouds():
    if False:
        i = 10
        return i + 15
    (X, Y) = get_clouds()
    (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=0.33)
    return (Xtrain, Xtest, Ytrain, Ytest, linear, 1e-05, 400)
if __name__ == '__main__':
    (Xtrain, Xtest, Ytrain, Ytest, kernel, lr, n_iters) = spiral()
    print('Possible labels:', set(Ytrain))
    Ytrain[Ytrain == 0] = -1
    Ytest[Ytest == 0] = -1
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    model = SVM(kernel=kernel, C=1.0)
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain, lr=lr, n_iters=n_iters)
    print('train duration:', datetime.now() - t0)
    t0 = datetime.now()
    print('train score:', model.score(Xtrain, Ytrain), 'duration:', datetime.now() - t0)
    t0 = datetime.now()
    print('test score:', model.score(Xtest, Ytest), 'duration:', datetime.now() - t0)
    if Xtrain.shape[1] == 2:
        plot_decision_boundary(model)