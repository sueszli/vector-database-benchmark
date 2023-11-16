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

def linear(X1, X2):
    if False:
        for i in range(10):
            print('nop')
    return X1.dot(X2.T)

def rbf(X1, X2):
    if False:
        print('Hello World!')
    gamma = 5.0
    if np.ndim(X1) == 1 and np.ndim(X2) == 1:
        result = np.exp(-gamma * np.linalg.norm(X1 - X2) ** 2)
    elif np.ndim(X1) > 1 and np.ndim(X2) == 1 or (np.ndim(X1) == 1 and np.ndim(X2) > 1):
        result = np.exp(-gamma * np.linalg.norm(X1 - X2, axis=1) ** 2)
    elif np.ndim(X1) > 1 and np.ndim(X2) > 1:
        result = np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) ** 2)
    return result

def sigmoid(X1, X2, gamma=0.05, c=1):
    if False:
        for i in range(10):
            print('nop')
    return np.tanh(gamma * X1.dot(X2.T) + c)

class SVM:

    def __init__(self, kernel, C=1.0):
        if False:
            for i in range(10):
                print('nop')
        self.kernel = kernel
        self.C = C

    def _loss(self, X, Y):
        if False:
            while True:
                i = 10
        return -np.sum(self.alphas) + 0.5 * np.sum(self.YYK * np.outer(self.alphas, self.alphas))

    def _take_step(self, i1, i2):
        if False:
            return 10
        if i1 == i2:
            return False
        alph1 = self.alphas[i1]
        alph2 = self.alphas[i2]
        y1 = self.Ytrain[i1]
        y2 = self.Ytrain[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]
        s = y1 * y2
        if y1 != y2:
            L = max(0, alph2 - alph1)
            H = min(self.C, self.C + alph2 - alph1)
        elif y1 == y2:
            L = max(0, alph1 + alph2 - self.C)
            H = min(self.C, alph1 + alph2)
        if L == H:
            return False
        k11 = self.kernel(self.Xtrain[i1], self.Xtrain[i1])
        k12 = self.kernel(self.Xtrain[i1], self.Xtrain[i2])
        k22 = self.kernel(self.Xtrain[i2], self.Xtrain[i2])
        eta = k11 + k22 - 2 * k12
        if eta > 0:
            a2 = alph2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            print('***** eta < 0 *****')
            alphas_i2 = self.alphas[i2]
            self.alphas[i2] = L
            Lobj = self._loss(self.Xtrain, self.Ytrain)
            self.alphas[i2] = H
            Hobj = self._loss(self.Xtrain, self.Ytrain)
            if Lobj < Hobj - self.eps:
                a2 = L
            elif Lobj > Hobj + self.eps:
                a2 = H
            else:
                a2 = alph2
            self.alphas[i2] = alphas_i2
        if a2 < 1e-08:
            a2 = 0.0
        elif a2 > self.C - 1e-08:
            a2 = self.C
        if np.abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
            return False
        a1 = alph1 + s * (alph2 - a2)
        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + self.b
        if 0 < a1 and a1 < self.C:
            b_new = b1
        elif 0 < a2 and a2 < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) * 0.5
        self.alphas[i1] = a1
        self.alphas[i2] = a2
        for (index, alph) in zip([i1, i2], [a1, a2]):
            if 0.0 < alph < self.C:
                self.errors[index] = 0.0
        non_opt = [n for n in range(self.N) if n != i1 and n != i2]
        self.errors[non_opt] = self.errors[non_opt] + y1 * (a1 - alph1) * self.kernel(self.Xtrain[i1], self.Xtrain[non_opt]) + y2 * (a2 - alph2) * self.kernel(self.Xtrain[i2], self.Xtrain[non_opt]) + self.b - b_new
        self.b = b_new
        return True

    def _examine_example(self, i2):
        if False:
            for i in range(10):
                print('nop')
        y2 = self.Ytrain[i2]
        alph2 = self.alphas[i2]
        E2 = self.errors[i2]
        r2 = E2 * y2
        if r2 < -self.tol and alph2 < self.C or (r2 > self.tol and alph2 > 0):
            if len(self.alphas[(self.alphas != 0) & (self.alphas != self.C)]) > 1:
                if self.errors[i2] > 0:
                    i1 = np.argmin(self.errors)
                elif self.errors[i2] <= 0:
                    i1 = np.argmax(self.errors)
                if self._take_step(i1, i2):
                    return 1
            for i1 in np.roll(np.where((self.alphas != 0) & (self.alphas != self.C))[0], np.random.choice(np.arange(self.N))):
                if self._take_step(i1, i2):
                    return 1
            for i1 in np.roll(np.arange(self.N), np.random.choice(np.arange(self.N))):
                if self._take_step(i1, i2):
                    return 1
        return 0

    def fit(self, X, Y, tol=1e-05, eps=0.01):
        if False:
            return 10
        self.tol = tol
        self.eps = eps
        self.Xtrain = X
        self.Ytrain = Y
        self.N = X.shape[0]
        self.alphas = np.zeros(self.N)
        self.b = 0.0
        self.errors = self._decision_function(self.Xtrain) - self.Ytrain
        self.K = self.kernel(X, X)
        self.YY = np.outer(Y, Y)
        self.YYK = self.K * self.YY
        iter_ = 0
        numChanged = 0
        examineAll = 1
        losses = []
        while numChanged > 0 or examineAll:
            print('iter:', iter_)
            iter_ += 1
            numChanged = 0
            if examineAll:
                for i in range(self.alphas.shape[0]):
                    examine_result = self._examine_example(i)
                    numChanged += examine_result
                    if examine_result:
                        loss = self._loss(self.Xtrain, self.Ytrain)
                        losses.append(loss)
            else:
                for i in np.where((self.alphas != 0) & (self.alphas != self.C))[0]:
                    examine_result = self._examine_example(i)
                    numChanged += examine_result
                    if examine_result:
                        loss = self._loss(self.Xtrain, self.Ytrain)
                        losses.append(loss)
            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1
        plt.plot(losses)
        plt.title('loss per iteration')
        plt.show()

    def _decision_function(self, X):
        if False:
            for i in range(10):
                print('nop')
        return (self.alphas * self.Ytrain).dot(self.kernel(self.Xtrain, X)) - self.b

    def predict(self, X):
        if False:
            print('Hello World!')
        return np.sign(self._decision_function(X))

    def score(self, X, Y):
        if False:
            return 10
        P = self.predict(X)
        return np.mean(Y == P)

def get_data():
    if False:
        for i in range(10):
            print('nop')
    (X, Y) = get_clouds()
    (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=0.33)
    return (Xtrain, Xtest, Ytrain, Ytest)
if __name__ == '__main__':
    (Xtrain, Xtest, Ytrain, Ytest) = get_data()
    print('Possible labels:', set(Ytrain))
    Ytrain[Ytrain == 0] = -1
    Ytest[Ytest == 0] = -1
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    model = SVM(kernel=linear)
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print('train duration:', datetime.now() - t0)
    t0 = datetime.now()
    print('train score:', model.score(Xtrain, Ytrain), 'duration:', datetime.now() - t0)
    t0 = datetime.now()
    print('test score:', model.score(Xtest, Ytest), 'duration:', datetime.now() - t0)
    if Xtrain.shape[1] == 2:
        plot_decision_boundary(model)