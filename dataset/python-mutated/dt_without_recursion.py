from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
from util import get_data, get_xor, get_donut
from datetime import datetime

def entropy(y):
    if False:
        i = 10
        return i + 15
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    return 1 - p0 * p0 - p1 * p1

class DecisionTree:

    def __init__(self, depth=0, max_depth=None):
        if False:
            while True:
                i = 10
        self.max_depth = max_depth
        self.root = {}

    def fit(self, X, Y):
        if False:
            for i in range(10):
                print('nop')
        current_node = self.root
        depth = 0
        queue = []
        while True:
            if len(Y) == 1 or len(set(Y)) == 1:
                current_node['col'] = None
                current_node['split'] = None
                current_node['left'] = None
                current_node['right'] = None
                current_node['prediction'] = Y[0]
            else:
                D = X.shape[1]
                cols = range(D)
                max_ig = 0
                best_col = None
                best_split = None
                for col in cols:
                    (ig, split) = self.find_split(X, Y, col)
                    if ig > max_ig:
                        max_ig = ig
                        best_col = col
                        best_split = split
                if max_ig == 0:
                    current_node['col'] = None
                    current_node['split'] = None
                    current_node['left'] = None
                    current_node['right'] = None
                    current_node['prediction'] = np.round(Y.mean())
                else:
                    current_node['col'] = best_col
                    current_node['split'] = best_split
                    if depth == self.max_depth:
                        current_node['left'] = None
                        current_node['right'] = None
                        current_node['prediction'] = [np.round(Y[X[:, best_col] < self.split].mean()), np.round(Y[X[:, best_col] >= self.split].mean())]
                    else:
                        left_idx = X[:, best_col] < best_split
                        Xleft = X[left_idx]
                        Yleft = Y[left_idx]
                        new_node = {}
                        current_node['left'] = new_node
                        left_data = {'node': new_node, 'X': Xleft, 'Y': Yleft}
                        queue.insert(0, left_data)
                        right_idx = X[:, best_col] >= best_split
                        Xright = X[right_idx]
                        Yright = Y[right_idx]
                        new_node = {}
                        current_node['right'] = new_node
                        right_data = {'node': new_node, 'X': Xright, 'Y': Yright}
                        queue.insert(0, right_data)
            if len(queue) == 0:
                break
            next_data = queue.pop()
            current_node = next_data['node']
            X = next_data['X']
            Y = next_data['Y']

    def find_split(self, X, Y, col):
        if False:
            for i in range(10):
                print('nop')
        x_values = X[:, col]
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx]
        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        best_split = None
        max_ig = 0
        last_ig = 0
        for b in boundaries:
            split = (x_values[b] + x_values[b + 1]) / 2
            ig = self.information_gain(x_values, y_values, split)
            if ig < last_ig:
                break
            last_ig = ig
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return (max_ig, best_split)

    def information_gain(self, x, y, split):
        if False:
            for i in range(10):
                print('nop')
        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0
        return entropy(y) - p0 * entropy(y0) - p1 * entropy(y1)

    def predict_one(self, x):
        if False:
            i = 10
            return i + 15
        p = None
        current_node = self.root
        while True:
            if current_node['col'] is not None and current_node['split'] is not None:
                feature = x[current_node['col']]
                if feature < current_node['split']:
                    if current_node['left']:
                        current_node = current_node['left']
                    else:
                        p = current_node['prediction'][0]
                        break
                elif current_node['right']:
                    current_node = current_node['right']
                else:
                    p = current_node['prediction'][1]
                    break
            else:
                p = current_node['prediction']
                break
        return p

    def predict(self, X):
        if False:
            print('Hello World!')
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P

    def score(self, X, Y):
        if False:
            for i in range(10):
                print('nop')
        P = self.predict(X)
        return np.mean(P == Y)
if __name__ == '__main__':
    (X, Y) = get_data()
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]
    Ntrain = len(Y) // 2
    (Xtrain, Ytrain) = (X[:Ntrain], Y[:Ntrain])
    (Xtest, Ytest) = (X[Ntrain:], Y[Ntrain:])
    model = DecisionTree()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print('Training time:', datetime.now() - t0)
    t0 = datetime.now()
    print('Train accuracy:', model.score(Xtrain, Ytrain))
    print('Time to compute train accuracy:', datetime.now() - t0)
    t0 = datetime.now()
    print('Test accuracy:', model.score(Xtest, Ytest))
    print('Time to compute test accuracy:', datetime.now() - t0)
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print('SK: Training time:', datetime.now() - t0)
    t0 = datetime.now()
    print('Train accuracy:', model.score(Xtrain, Ytrain))
    print('SK: Time to compute train accuracy:', datetime.now() - t0)
    t0 = datetime.now()
    print('Test accuracy:', model.score(Xtest, Ytest))
    print('SK: Time to compute test accuracy:', datetime.now() - t0)