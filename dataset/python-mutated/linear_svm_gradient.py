from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from util import get_clouds
import numpy as np
import matplotlib.pyplot as plt

class LinearSVM:

    def __init__(self, C=1.0):
        if False:
            print('Hello World!')
        self.C = C

    def _objective(self, margins):
        if False:
            while True:
                i = 10
        return 0.5 * self.w.dot(self.w) + self.C * np.maximum(0, 1 - margins).sum()

    def fit(self, X, Y, lr=1e-05, n_iters=400):
        if False:
            print('Hello World!')
        (N, D) = X.shape
        self.N = N
        self.w = np.random.randn(D)
        self.b = 0
        losses = []
        for _ in range(n_iters):
            margins = Y * self._decision_function(X)
            loss = self._objective(margins)
            losses.append(loss)
            idx = np.where(margins < 1)[0]
            grad_w = self.w - self.C * Y[idx].dot(X[idx])
            self.w -= lr * grad_w
            grad_b = -self.C * Y[idx].sum()
            self.b -= lr * grad_b
        self.support_ = np.where(Y * self._decision_function(X) <= 1)[0]
        print('num SVs:', len(self.support_))
        print('w:', self.w)
        print('b:', self.b)
        plt.plot(losses)
        plt.title('loss per iteration')
        plt.show()

    def _decision_function(self, X):
        if False:
            print('Hello World!')
        return X.dot(self.w) + self.b

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        return np.sign(self._decision_function(X))

    def score(self, X, Y):
        if False:
            for i in range(10):
                print('nop')
        P = self.predict(X)
        return np.mean(Y == P)

def plot_decision_boundary(model, X, Y, resolution=100, colors=('b', 'k', 'r')):
    if False:
        while True:
            i = 10
    np.warnings.filterwarnings('ignore')
    (fig, ax) = plt.subplots()
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    grid = [[model._decision_function(np.array([[xr, yr]])) for yr in y_range] for xr in x_range]
    grid = np.array(grid).reshape(len(x_range), len(y_range))
    ax.contour(x_range, y_range, grid.T, (-1, 0, 1), linewidths=(1, 1, 1), linestyles=('--', '-', '--'), colors=colors)
    ax.scatter(X[:, 0], X[:, 1], c=Y, lw=0, alpha=0.3, cmap='seismic')
    mask = model.support_
    ax.scatter(X[:, 0][mask], X[:, 1][mask], c=Y[mask], cmap='seismic')
    ax.scatter([0], [0], c='black', marker='x')
    plt.show()

def clouds():
    if False:
        for i in range(10):
            print('nop')
    (X, Y) = get_clouds()
    (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=0.33)
    return (Xtrain, Xtest, Ytrain, Ytest, 0.001, 200)

def medical():
    if False:
        i = 10
        return i + 15
    data = load_breast_cancer()
    (X, Y) = (data.data, data.target)
    (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=0.33)
    return (Xtrain, Xtest, Ytrain, Ytest, 0.001, 200)
if __name__ == '__main__':
    (Xtrain, Xtest, Ytrain, Ytest, lr, n_iters) = clouds()
    print('Possible labels:', set(Ytrain))
    Ytrain[Ytrain == 0] = -1
    Ytest[Ytest == 0] = -1
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    model = LinearSVM(C=1.0)
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain, lr=lr, n_iters=n_iters)
    print('train duration:', datetime.now() - t0)
    t0 = datetime.now()
    print('train score:', model.score(Xtrain, Ytrain), 'duration:', datetime.now() - t0)
    t0 = datetime.now()
    print('test score:', model.score(Xtest, Ytest), 'duration:', datetime.now() - t0)
    if Xtrain.shape[1] == 2:
        plot_decision_boundary(model, Xtrain, Ytrain)