from __future__ import annotations
import copy
import itertools
import math
import random
import numpy as np
import pandas as pd
import pytest
from sklearn import linear_model as sklm
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import log_loss
from river import datasets, optim, preprocessing, stream, utils
from river import linear_model as lm

def iter_perturbations(keys, n=10):
    if False:
        print('Hello World!')
    'Enumerate perturbations that will be applied to the weights.'
    for i in keys:
        yield {j: int(i == j) for j in keys}
    for _ in range(n):
        p = {j: random.gauss(0, 1) for j in keys}
        norm = utils.math.norm(p, order=2)
        for j in p:
            p[j] /= norm
        yield p

@pytest.mark.parametrize('lm, dataset', [pytest.param(lm(optimizer=copy.deepcopy(optimizer), initializer=initializer, l2=0), dataset, id=f'{lm.__name__} - {optimizer} - {initializer}') for (lm, dataset) in [(lm.LinearRegression, datasets.TrumpApproval().take(100)), (lm.LogisticRegression, datasets.Bananas().take(100))] for (optimizer, initializer) in itertools.product([optim.AdaBound(), optim.AdaDelta(), optim.AdaGrad(), optim.AdaMax(), optim.Adam(), optim.AMSGrad(), optim.RMSProp(), optim.SGD()], [optim.initializers.Zeros(), optim.initializers.Normal(mu=0, sigma=1, seed=42)])])
def test_finite_differences(lm, dataset):
    if False:
        return 10
    'Checks the gradient of a linear model via finite differences.\n\n    References\n    ----------\n    [^1]: [How to test gradient implementations](https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/)\n    [^2]: [Stochastic Gradient Descent Tricks](https://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf)\n\n    '
    scaler = preprocessing.StandardScaler()
    eps = 1e-06
    for (x, y) in dataset:
        x = scaler.learn_one(x).transform_one(x)
        (gradient, _) = lm._eval_gradient_one(x, y, 1)
        weights = copy.deepcopy(lm._weights)
        for d in iter_perturbations(weights.keys()):
            lm._weights = utils.VectorDict({i: weights[i] + eps * di for (i, di) in d.items()})
            forward = lm.loss(y_true=y, y_pred=lm._raw_dot_one(x))
            lm._weights = utils.VectorDict({i: weights[i] - eps * di for (i, di) in d.items()})
            backward = lm.loss(y_true=y, y_pred=lm._raw_dot_one(x))
            g = utils.math.dot(d, gradient)
            h = (forward - backward) / (2 * eps)
            assert abs(g - h) < 1e-05
        lm._weights = weights
        lm.learn_one(x, y)

def test_one_many_consistent():
    if False:
        i = 10
        return i + 15
    'Checks that using learn_one or learn_many produces the same result.'
    X = pd.read_csv(datasets.TrumpApproval().path)
    Y = X.pop('five_thirty_eight')
    one = lm.LinearRegression()
    for (x, y) in stream.iter_pandas(X[:100], Y[:100]):
        one.learn_one(x, y)
    many = lm.LinearRegression()
    for (xb, yb) in zip(np.array_split(X[:100], len(X[:100])), np.array_split(Y[:100], len(Y[:100]))):
        many.learn_many(xb, yb)
    for i in X:
        assert math.isclose(one.weights[i], many.weights[i])

def test_shuffle_columns():
    if False:
        for i in range(10):
            print('nop')
    'Checks that learn_many works identically whether columns are shuffled or not.'
    X = pd.read_csv(datasets.TrumpApproval().path)
    Y = X.pop('five_thirty_eight')
    normal = lm.LinearRegression()
    for (xb, yb) in zip(np.array_split(X, 10), np.array_split(Y, 10)):
        normal.learn_many(xb, yb)
    shuffled = lm.LinearRegression()
    for (xb, yb) in zip(np.array_split(X, 10), np.array_split(Y, 10)):
        cols = np.random.permutation(X.columns)
        shuffled.learn_many(xb[cols], yb)
    for i in X:
        assert math.isclose(normal.weights[i], shuffled.weights[i])

def test_add_remove_columns():
    if False:
        print('Hello World!')
    'Checks that no exceptions are raised whenever columns are dropped and/or added.'
    X = pd.read_csv(datasets.TrumpApproval().path)
    Y = X.pop('five_thirty_eight')
    lin_reg = lm.LinearRegression()
    for (xb, yb) in zip(np.array_split(X, 10), np.array_split(Y, 10)):
        cols = np.random.choice(X.columns, len(X.columns) // 2, replace=False)
        lin_reg.learn_many(xb[cols], yb)

class ScikitLearnSquaredLoss:
    """sklearn removes the leading 2 from the gradient of the squared loss."""

    def gradient(self, y_true, y_pred):
        if False:
            print('Hello World!')
        return y_pred - y_true
lin_reg_tests = {'Vanilla': ({'optimizer': optim.SGD(0.01), 'loss': ScikitLearnSquaredLoss()}, {'learning_rate': 'constant', 'eta0': 0.01, 'alpha': 0}), 'Huber': ({'optimizer': optim.SGD(0.01), 'loss': optim.losses.Huber()}, {'loss': 'huber', 'learning_rate': 'constant', 'eta0': 0.01, 'alpha': 0}), 'No intercept': ({'optimizer': optim.SGD(0.01), 'intercept_lr': 0, 'loss': ScikitLearnSquaredLoss()}, {'learning_rate': 'constant', 'eta0': 0.01, 'alpha': 0, 'fit_intercept': False}), 'L2 regu': ({'optimizer': optim.SGD(0.01), 'loss': ScikitLearnSquaredLoss(), 'l2': 0.001}, {'learning_rate': 'constant', 'eta0': 0.01, 'alpha': 0.001})}

@pytest.mark.parametrize('river_params, sklearn_params', lin_reg_tests.values(), ids=lin_reg_tests.keys())
def test_lin_reg_sklearn_coherence(river_params, sklearn_params):
    if False:
        print('Hello World!')
    'Checks that the sklearn and river implementations produce the same results.'
    ss = preprocessing.StandardScaler()
    rv = lm.LinearRegression(**river_params)
    sk = sklm.SGDRegressor(**sklearn_params)
    for (x, y) in datasets.TrumpApproval().take(100):
        x = ss.learn_one(x).transform_one(x)
        rv.learn_one(x, y)
        sk.partial_fit([list(x.values())], [y])
    for (i, w) in enumerate(rv.weights.values()):
        assert math.isclose(w, sk.coef_[i])
    assert math.isclose(rv.intercept, sk.intercept_[0])

@pytest.mark.parametrize('river_params, sklearn_params', lin_reg_tests.values(), ids=lin_reg_tests.keys())
def test_lin_reg_sklearn_learn_many_coherence(river_params, sklearn_params):
    if False:
        return 10
    'Checks that the sklearn and river implementations produce the same results\n    when learn_many is used.'
    ss = preprocessing.StandardScaler()
    rv = lm.LinearRegression(**river_params)
    sk = sklm.SGDRegressor(**sklearn_params)
    for (x, y) in datasets.TrumpApproval().take(100):
        x = ss.learn_one(x).transform_one(x)
        rv.learn_many(pd.DataFrame([x]), pd.Series([y]))
        sk.partial_fit([list(x.values())], [y])
    for (i, w) in enumerate(rv.weights.values()):
        assert math.isclose(w, sk.coef_[i])
    assert math.isclose(rv.intercept, sk.intercept_[0])
log_reg_tests = {'Vanilla': ({'optimizer': optim.SGD(0.01)}, {'learning_rate': 'constant', 'eta0': 0.01, 'alpha': 0, 'loss': 'log_loss'}), 'Hinge': ({'optimizer': optim.SGD(0.01), 'loss': optim.losses.Hinge()}, {'learning_rate': 'constant', 'eta0': 0.01, 'alpha': 0}), 'No intercept': ({'optimizer': optim.SGD(0.01), 'intercept_lr': 0}, {'learning_rate': 'constant', 'eta0': 0.01, 'alpha': 0, 'loss': 'log_loss', 'fit_intercept': False}), 'L2 regu': ({'optimizer': optim.SGD(0.01), 'l2': 0.001}, {'learning_rate': 'constant', 'eta0': 0.01, 'alpha': 0.001, 'loss': 'log_loss'}), 'Inverse-scaling': ({'optimizer': optim.SGD(optim.schedulers.InverseScaling(0.01)), 'intercept_lr': optim.schedulers.InverseScaling(0.01)}, {'eta0': 0.01, 'alpha': 0, 'learning_rate': 'invscaling', 'loss': 'log_loss'}), 'Optimal': ({'optimizer': optim.SGD(optim.schedulers.Optimal(optim.losses.Hinge(), alpha=0.001)), 'loss': optim.losses.Hinge(), 'intercept_lr': optim.schedulers.Optimal(optim.losses.Hinge(), alpha=0.001), 'l2': 0.001}, {'learning_rate': 'optimal', 'alpha': 0.001}), 'Optimal no intercept': ({'optimizer': optim.SGD(optim.schedulers.Optimal(optim.losses.Hinge(), alpha=0.001)), 'loss': optim.losses.Hinge(), 'intercept_lr': 0, 'l2': 0.001}, {'learning_rate': 'optimal', 'alpha': 0.001, 'fit_intercept': False})}

@pytest.mark.parametrize('river_params, sklearn_params', log_reg_tests.values(), ids=log_reg_tests.keys())
def test_log_reg_sklearn_coherence(river_params, sklearn_params):
    if False:
        i = 10
        return i + 15
    'Checks that the sklearn and river implementations produce the same results.'
    ss = preprocessing.StandardScaler()
    rv = lm.LogisticRegression(**river_params)
    sk = sklm.SGDClassifier(**sklearn_params)
    for (x, y) in datasets.Bananas().take(100):
        x = ss.learn_one(x).transform_one(x)
        rv.learn_one(x, y)
        sk.partial_fit([list(x.values())], [y], classes=[False, True])
    for (i, w) in enumerate(rv.weights.values()):
        assert math.isclose(w, sk.coef_[0][i])
    assert math.isclose(rv.intercept, sk.intercept_[0])
perceptron_tests = {'Vanilla': ({}, {}), 'L2 regu': ({'l2': 0.001}, {'alpha': 0.001, 'penalty': 'l2'})}

@pytest.mark.parametrize('river_params, sklearn_params', perceptron_tests.values(), ids=perceptron_tests.keys())
def test_perceptron_sklearn_coherence(river_params, sklearn_params):
    if False:
        print('Hello World!')
    'Checks that the sklearn and river implementations produce the same results.'
    ss = preprocessing.StandardScaler()
    rv = lm.Perceptron(**river_params)
    sk = sklm.Perceptron(**sklearn_params)
    for (x, y) in datasets.Bananas().take(100):
        x = ss.learn_one(x).transform_one(x)
        rv.learn_one(x, y)
        sk.partial_fit([list(x.values())], [y], classes=[False, True])
    for (i, w) in enumerate(rv.weights.values()):
        assert math.isclose(w, sk.coef_[0][i])
    assert math.isclose(rv.intercept, sk.intercept_[0])

def test_lin_reg_sklearn_l1_non_regression():
    if False:
        while True:
            i = 10
    'Checks that the river L1 implementation results are no worse than sklearn L1.'
    (X, y, true_coeffs) = make_regression(n_samples=1000, n_features=20, n_informative=4, coef=True, random_state=273)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    ss = preprocessing.StandardScaler()
    rv = lm.LinearRegression(**{'optimizer': optim.SGD(0.01), 'loss': ScikitLearnSquaredLoss(), 'l1': 0.1})
    sk = sklm.SGDRegressor(**{'learning_rate': 'constant', 'eta0': 0.01, 'alpha': 0.1, 'penalty': 'l1'})
    for (xi, yi) in stream.iter_pandas(X, y):
        xi_tr = ss.learn_one(xi).transform_one(xi)
        rv.learn_one(xi_tr, yi)
        sk.partial_fit([list(xi_tr.values())], [yi])
    rv_coeffs = np.array(list(rv.weights.values()))
    sk_coeffs = sk.coef_
    assert np.sum(rv_coeffs > 0) <= np.sum(sk_coeffs > 0)
    assert np.isclose(rv_coeffs, true_coeffs, rtol=0.05, atol=0.0).all()

def test_log_reg_sklearn_l1_non_regression():
    if False:
        print('Hello World!')
    'Checks that the river L1 implementation results are no worse than sklearn L1.'
    (X, y) = make_classification(n_samples=1000, n_features=20, n_informative=4, n_classes=2, random_state=273)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    ss = preprocessing.StandardScaler()
    rv = lm.LogisticRegression(**{'optimizer': optim.SGD(0.01), 'l1': 0.001})
    sk = sklm.SGDClassifier(**{'learning_rate': 'constant', 'eta0': 0.01, 'alpha': 0.001, 'penalty': 'l1', 'loss': 'log_loss'})
    rv_pred = list()
    sk_pred = list()
    for (xi, yi) in stream.iter_pandas(X, y):
        xi_tr = ss.learn_one(xi).transform_one(xi)
        rv.learn_one(xi_tr, yi)
        sk.partial_fit([list(xi_tr.values())], [yi], classes=[False, True])
        rv_pred.append(rv.predict_one(xi_tr))
        sk_pred.append(sk.predict([list(xi_tr.values())])[0])
    rv_coeffs = np.array(list(rv.weights.values()))
    sk_coeffs = sk.coef_
    assert np.sum(rv_coeffs > 0) <= np.sum(sk_coeffs > 0)
    assert math.isclose(log_loss(y, rv_pred), log_loss(y, sk_pred))