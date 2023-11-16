"""
Testing for Theil-Sen module (sklearn.linear_model.theil_sen)
"""
import os
import re
import sys
from contextlib import contextmanager
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_array_less
from scipy.linalg import norm
from scipy.optimize import fmin_bfgs
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model._theil_sen import _breakdown_point, _modified_weiszfeld_step, _spatial_median
from sklearn.utils._testing import assert_almost_equal

@contextmanager
def no_stdout_stderr():
    if False:
        i = 10
        return i + 15
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
        devnull.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def gen_toy_problem_1d(intercept=True):
    if False:
        for i in range(10):
            print('nop')
    random_state = np.random.RandomState(0)
    w = 3.0
    if intercept:
        c = 2.0
        n_samples = 50
    else:
        c = 0.1
        n_samples = 100
    x = random_state.normal(size=n_samples)
    noise = 0.1 * random_state.normal(size=n_samples)
    y = w * x + c + noise
    if intercept:
        (x[42], y[42]) = (-2, 4)
        (x[43], y[43]) = (-2.5, 8)
        (x[33], y[33]) = (2.5, 1)
        (x[49], y[49]) = (2.1, 2)
    else:
        (x[42], y[42]) = (-2, 4)
        (x[43], y[43]) = (-2.5, 8)
        (x[53], y[53]) = (2.5, 1)
        (x[60], y[60]) = (2.1, 2)
        (x[72], y[72]) = (1.8, -7)
    return (x[:, np.newaxis], y, w, c)

def gen_toy_problem_2d():
    if False:
        for i in range(10):
            print('nop')
    random_state = np.random.RandomState(0)
    n_samples = 100
    X = random_state.normal(size=(n_samples, 2))
    w = np.array([5.0, 10.0])
    c = 1.0
    noise = 0.1 * random_state.normal(size=n_samples)
    y = np.dot(X, w) + c + noise
    n_outliers = n_samples // 10
    ix = random_state.randint(0, n_samples, size=n_outliers)
    y[ix] = 50 * random_state.normal(size=n_outliers)
    return (X, y, w, c)

def gen_toy_problem_4d():
    if False:
        for i in range(10):
            print('nop')
    random_state = np.random.RandomState(0)
    n_samples = 10000
    X = random_state.normal(size=(n_samples, 4))
    w = np.array([5.0, 10.0, 42.0, 7.0])
    c = 1.0
    noise = 0.1 * random_state.normal(size=n_samples)
    y = np.dot(X, w) + c + noise
    n_outliers = n_samples // 10
    ix = random_state.randint(0, n_samples, size=n_outliers)
    y[ix] = 50 * random_state.normal(size=n_outliers)
    return (X, y, w, c)

def test_modweiszfeld_step_1d():
    if False:
        print('Hello World!')
    X = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
    median = 2.0
    new_y = _modified_weiszfeld_step(X, median)
    assert_array_almost_equal(new_y, median)
    y = 2.5
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_less(median, new_y)
    assert_array_less(new_y, y)
    y = 3.0
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_less(median, new_y)
    assert_array_less(new_y, y)
    X = np.array([1.0, 2.0, 3.0]).reshape(1, 3)
    y = X[0]
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_equal(y, new_y)

def test_modweiszfeld_step_2d():
    if False:
        while True:
            i = 10
    X = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]).reshape(3, 2)
    y = np.array([0.5, 0.5])
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_almost_equal(new_y, np.array([1 / 3, 2 / 3]))
    new_y = _modified_weiszfeld_step(X, new_y)
    assert_array_almost_equal(new_y, np.array([0.2792408, 0.7207592]))
    y = np.array([0.21132505, 0.78867497])
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_almost_equal(new_y, y)

def test_spatial_median_1d():
    if False:
        i = 10
        return i + 15
    X = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
    true_median = 2.0
    (_, median) = _spatial_median(X)
    assert_array_almost_equal(median, true_median)
    random_state = np.random.RandomState(0)
    X = random_state.randint(100, size=(1000, 1))
    true_median = np.median(X.ravel())
    (_, median) = _spatial_median(X)
    assert_array_equal(median, true_median)

def test_spatial_median_2d():
    if False:
        i = 10
        return i + 15
    X = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]).reshape(3, 2)
    (_, median) = _spatial_median(X, max_iter=100, tol=1e-06)

    def cost_func(y):
        if False:
            while True:
                i = 10
        dists = np.array([norm(x - y) for x in X])
        return np.sum(dists)
    fermat_weber = fmin_bfgs(cost_func, median, disp=False)
    assert_array_almost_equal(median, fermat_weber)
    warning_message = 'Maximum number of iterations 30 reached in spatial median.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        _spatial_median(X, max_iter=30, tol=0.0)

def test_theil_sen_1d():
    if False:
        for i in range(10):
            print('nop')
    (X, y, w, c) = gen_toy_problem_1d()
    lstq = LinearRegression().fit(X, y)
    assert np.abs(lstq.coef_ - w) > 0.9
    theil_sen = TheilSenRegressor(random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    assert_array_almost_equal(theil_sen.intercept_, c, 1)

def test_theil_sen_1d_no_intercept():
    if False:
        for i in range(10):
            print('nop')
    (X, y, w, c) = gen_toy_problem_1d(intercept=False)
    lstq = LinearRegression(fit_intercept=False).fit(X, y)
    assert np.abs(lstq.coef_ - w - c) > 0.5
    theil_sen = TheilSenRegressor(fit_intercept=False, random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w + c, 1)
    assert_almost_equal(theil_sen.intercept_, 0.0)
    theil_sen.score(X, y)

def test_theil_sen_2d():
    if False:
        i = 10
        return i + 15
    (X, y, w, c) = gen_toy_problem_2d()
    lstq = LinearRegression().fit(X, y)
    assert norm(lstq.coef_ - w) > 1.0
    theil_sen = TheilSenRegressor(max_subpopulation=1000.0, random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    assert_array_almost_equal(theil_sen.intercept_, c, 1)

def test_calc_breakdown_point():
    if False:
        while True:
            i = 10
    bp = _breakdown_point(10000000000.0, 2)
    assert np.abs(bp - 1 + 1 / np.sqrt(2)) < 1e-06

@pytest.mark.parametrize('param, ExceptionCls, match', [({'n_subsamples': 1}, ValueError, re.escape('Invalid parameter since n_features+1 > n_subsamples (2 > 1)')), ({'n_subsamples': 101}, ValueError, re.escape('Invalid parameter since n_subsamples > n_samples (101 > 50)'))])
def test_checksubparams_invalid_input(param, ExceptionCls, match):
    if False:
        return 10
    (X, y, w, c) = gen_toy_problem_1d()
    theil_sen = TheilSenRegressor(**param, random_state=0)
    with pytest.raises(ExceptionCls, match=match):
        theil_sen.fit(X, y)

def test_checksubparams_n_subsamples_if_less_samples_than_features():
    if False:
        for i in range(10):
            print('nop')
    random_state = np.random.RandomState(0)
    (n_samples, n_features) = (10, 20)
    X = random_state.normal(size=(n_samples, n_features))
    y = random_state.normal(size=n_samples)
    theil_sen = TheilSenRegressor(n_subsamples=9, random_state=0)
    with pytest.raises(ValueError):
        theil_sen.fit(X, y)

def test_subpopulation():
    if False:
        for i in range(10):
            print('nop')
    (X, y, w, c) = gen_toy_problem_4d()
    theil_sen = TheilSenRegressor(max_subpopulation=250, random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    assert_array_almost_equal(theil_sen.intercept_, c, 1)

def test_subsamples():
    if False:
        i = 10
        return i + 15
    (X, y, w, c) = gen_toy_problem_4d()
    theil_sen = TheilSenRegressor(n_subsamples=X.shape[0], random_state=0).fit(X, y)
    lstq = LinearRegression().fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, lstq.coef_, 9)

def test_verbosity():
    if False:
        return 10
    (X, y, w, c) = gen_toy_problem_1d()
    with no_stdout_stderr():
        TheilSenRegressor(verbose=True, random_state=0).fit(X, y)
        TheilSenRegressor(verbose=True, max_subpopulation=10, random_state=0).fit(X, y)

def test_theil_sen_parallel():
    if False:
        return 10
    (X, y, w, c) = gen_toy_problem_2d()
    lstq = LinearRegression().fit(X, y)
    assert norm(lstq.coef_ - w) > 1.0
    theil_sen = TheilSenRegressor(n_jobs=2, random_state=0, max_subpopulation=2000.0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    assert_array_almost_equal(theil_sen.intercept_, c, 1)

def test_less_samples_than_features():
    if False:
        print('Hello World!')
    random_state = np.random.RandomState(0)
    (n_samples, n_features) = (10, 20)
    X = random_state.normal(size=(n_samples, n_features))
    y = random_state.normal(size=n_samples)
    theil_sen = TheilSenRegressor(fit_intercept=False, random_state=0).fit(X, y)
    lstq = LinearRegression(fit_intercept=False).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, lstq.coef_, 12)
    theil_sen = TheilSenRegressor(fit_intercept=True, random_state=0).fit(X, y)
    y_pred = theil_sen.predict(X)
    assert_array_almost_equal(y_pred, y, 12)