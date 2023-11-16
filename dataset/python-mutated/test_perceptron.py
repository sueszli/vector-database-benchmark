import sys
import numpy as np
from sklearn.base import clone
from mlxtend.classifier import Perceptron
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises
(X, y) = iris_data()
X = X[:, [0, 3]]
X = X[0:100]
y0 = y[0:100]
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

def test_invalid_labels_1():
    if False:
        while True:
            i = 10
    y1 = np.where(y0 == 0, 2, 1)
    ppn = Perceptron(epochs=15, eta=0.01, random_seed=1)
    if sys.version_info >= (3, 0):
        objtype = '{(0, 1)}'
    else:
        objtype = 'set([(0, 1)])'
    expect = 'Labels not in %s.\nFound (1, 2)' % objtype
    assert_raises(AttributeError, expect, ppn.fit, X, y1, {(0, 1)})

def test_invalid_labels_2():
    if False:
        return 10
    y1 = np.where(y0 == 0, -1, 1)
    ppn = Perceptron(epochs=15, eta=0.01, random_seed=1)
    assert_raises(AttributeError, 'y array must not contain negative labels.\nFound [-1  1]', ppn.fit, X, y1, {(-1, 1)})

def test_standardized_iris_data():
    if False:
        while True:
            i = 10
    ppn = Perceptron(epochs=15, eta=0.01, random_seed=1)
    ppn = ppn.fit(X_std, y0)
    assert (y0 == ppn.predict(X_std)).all(), ppn.predict(X_std)

def test_progress_1():
    if False:
        return 10
    ppn = Perceptron(epochs=15, eta=0.01, random_seed=1, print_progress=1)
    ppn = ppn.fit(X_std, y0)

def test_progress_2():
    if False:
        for i in range(10):
            print('nop')
    ppn = Perceptron(epochs=15, eta=0.01, random_seed=1, print_progress=2)
    ppn = ppn.fit(X_std, y0)

def test_progress_3():
    if False:
        return 10
    ppn = Perceptron(epochs=15, eta=0.01, random_seed=1, print_progress=3)
    ppn = ppn.fit(X_std, y0)

def test_score_function():
    if False:
        for i in range(10):
            print('nop')
    ppn = Perceptron(epochs=15, eta=0.01, random_seed=1)
    ppn = ppn.fit(X_std, y0)
    acc = ppn.score(X_std, y0)
    assert acc == 1.0, acc

def test_nonstandardized_iris_data():
    if False:
        print('Hello World!')
    ppn = Perceptron(epochs=100, eta=0.01, random_seed=1)
    ppn = ppn.fit(X, y0)
    assert (y0 == ppn.predict(X)).all()

def test_clone():
    if False:
        for i in range(10):
            print('nop')
    ppn = Perceptron()
    clone(ppn)