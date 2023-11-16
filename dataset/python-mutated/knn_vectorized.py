from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances

class KNN(object):

    def __init__(self, k):
        if False:
            print('Hello World!')
        self.k = k

    def fit(self, X, y):
        if False:
            print('Hello World!')
        self.X = X
        self.y = y

    def predict(self, X):
        if False:
            return 10
        N = len(X)
        y = np.zeros(N)
        distances = pairwise_distances(X, self.X)
        idx = distances.argsort(axis=1)[:, :self.k]
        votes = self.y[idx]
        for i in range(N):
            y[i] = np.bincount(votes[i]).argmax()
        return y

    def score(self, X, Y):
        if False:
            for i in range(10):
                print('nop')
        P = self.predict(X)
        return np.mean(P == Y)
if __name__ == '__main__':
    (X, Y) = get_data(2000)
    Ntrain = 1000
    (Xtrain, Ytrain) = (X[:Ntrain], Y[:Ntrain])
    (Xtest, Ytest) = (X[Ntrain:], Y[Ntrain:])
    train_scores = []
    test_scores = []
    ks = (1, 2, 3, 4, 5)
    for k in ks:
        print('\nk =', k)
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print('Training time:', datetime.now() - t0)
        t0 = datetime.now()
        train_score = knn.score(Xtrain, Ytrain)
        train_scores.append(train_score)
        print('Train accuracy:', train_score)
        print('Time to compute train accuracy:', datetime.now() - t0, 'Train size:', len(Ytrain))
        t0 = datetime.now()
        test_score = knn.score(Xtest, Ytest)
        print('Test accuracy:', test_score)
        test_scores.append(test_score)
        print('Time to compute test accuracy:', datetime.now() - t0, 'Test size:', len(Ytest))
    plt.plot(ks, train_scores, label='train scores')
    plt.plot(ks, test_scores, label='test scores')
    plt.legend()
    plt.show()