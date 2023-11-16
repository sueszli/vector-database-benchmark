import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.dimred import SlicedInverseReg, SAVE, PHD, CORE
from numpy.testing import assert_equal, assert_allclose
from statsmodels.tools.numdiff import approx_fprime

def test_poisson():
    if False:
        while True:
            i = 10
    np.random.seed(43242)
    xmat = np.random.normal(size=(500, 5))
    xmat[:, 1] = 0.5 * xmat[:, 0] + np.sqrt(1 - 0.5 ** 2) * xmat[:, 1]
    xmat[:, 3] = 0.5 * xmat[:, 2] + np.sqrt(1 - 0.5 ** 2) * xmat[:, 3]
    b = np.r_[0, 1, -1, 0, 0.5]
    lpr = np.dot(xmat, b)
    ev = np.exp(lpr)
    y = np.random.poisson(ev)
    for method in range(6):
        if method == 0:
            model = SlicedInverseReg(y, xmat)
            rslt = model.fit()
        elif method == 1:
            model = SAVE(y, xmat)
            rslt = model.fit(slice_n=100)
        elif method == 2:
            model = SAVE(y, xmat, bc=True)
            rslt = model.fit(slice_n=100)
        elif method == 3:
            df = pd.DataFrame({'y': y, 'x0': xmat[:, 0], 'x1': xmat[:, 1], 'x2': xmat[:, 2], 'x3': xmat[:, 3], 'x4': xmat[:, 4]})
            model = SlicedInverseReg.from_formula('y ~ 0 + x0 + x1 + x2 + x3 + x4', data=df)
            rslt = model.fit()
        elif method == 4:
            model = PHD(y, xmat)
            rslt = model.fit()
        elif method == 5:
            model = PHD(y, xmat)
            rslt = model.fit(resid=True)
        assert_equal(np.abs(rslt.eigs[0] / rslt.eigs[1]) > 5, True)
        params = np.asarray(rslt.params)
        q = np.dot(params[:, 0], b)
        q /= np.sqrt(np.sum(params[:, 0] ** 2))
        q /= np.sqrt(np.sum(b ** 2))
        assert_equal(np.abs(q) > 0.95, True)

def test_sir_regularized_numdiff():
    if False:
        while True:
            i = 10
    np.random.seed(93482)
    n = 1000
    p = 10
    xmat = np.random.normal(size=(n, p))
    y1 = np.dot(xmat, np.linspace(-1, 1, p))
    y2 = xmat.sum(1)
    y = y2 / (1 + y1 ** 2) + np.random.normal(size=n)
    model = SlicedInverseReg(y, xmat)
    _ = model.fit()
    fmat = np.zeros((p - 2, p))
    for i in range(p - 2):
        fmat[i, i:i + 3] = [1, -2, 1]
    with pytest.warns(UserWarning, match='SIR.fit_regularized did not'):
        _ = model.fit_regularized(2, 3 * fmat)
    for _ in range(5):
        pa = np.random.normal(size=(p, 2))
        (pa, _, _) = np.linalg.svd(pa, 0)
        gn = approx_fprime(pa.ravel(), model._regularized_objective, 1e-07)
        gr = model._regularized_grad(pa.ravel())
        assert_allclose(gn, gr, atol=1e-05, rtol=0.0001)

def test_sir_regularized_1d():
    if False:
        print('Hello World!')
    np.random.seed(93482)
    n = 1000
    p = 10
    xmat = np.random.normal(size=(n, p))
    y = np.dot(xmat[:, 0:4], np.r_[1, 1, -1, -1]) + np.random.normal(size=n)
    model = SlicedInverseReg(y, xmat)
    rslt = model.fit()
    fmat = np.zeros((2, p))
    fmat[0, 0:2] = [1, -1]
    fmat[1, 2:4] = [1, -1]
    rslt2 = model.fit_regularized(1, 3 * fmat)
    pa0 = np.zeros(p)
    pa0[0:4] = [1, 1, -1, -1]
    pa1 = rslt.params[:, 0]
    pa2 = rslt2.params[:, 0:2]

    def sim(x, y):
        if False:
            i = 10
            return i + 15
        x = x / np.sqrt(np.sum(x * x))
        y = y / np.sqrt(np.sum(y * y))
        return 1 - np.abs(np.dot(x, y))
    assert_equal(sim(pa0, pa1) > sim(pa0, pa2), True)
    assert_equal(sim(pa0, pa2) < 0.001, True)
    assert_equal(np.sum(np.dot(fmat, pa1) ** 2) > np.sum(np.dot(fmat, pa2) ** 2), True)

def test_sir_regularized_2d():
    if False:
        return 10
    np.random.seed(93482)
    n = 1000
    p = 10
    xmat = np.random.normal(size=(n, p))
    y1 = np.dot(xmat[:, 0:4], np.r_[1, 1, -1, -1])
    y2 = np.dot(xmat[:, 4:8], np.r_[1, 1, -1, -1])
    y = y1 + np.arctan(y2) + np.random.normal(size=n)
    model = SlicedInverseReg(y, xmat)
    rslt1 = model.fit()
    fmat = np.zeros((1, p))
    for d in (1, 2, 3, 4):
        if d < 3:
            rslt2 = model.fit_regularized(d, fmat)
        else:
            with pytest.warns(UserWarning, match='SIR.fit_regularized did'):
                rslt2 = model.fit_regularized(d, fmat)
        pa1 = rslt1.params[:, 0:d]
        (pa1, _, _) = np.linalg.svd(pa1, 0)
        pa2 = rslt2.params
        (_, s, _) = np.linalg.svd(np.dot(pa1.T, pa2))
        assert_allclose(np.sum(s), d, atol=0.1, rtol=0.1)

def test_covreduce():
    if False:
        while True:
            i = 10
    np.random.seed(34324)
    p = 4
    endog = []
    exog = []
    for k in range(3):
        c = np.eye(p)
        x = np.random.normal(size=(2, 2))
        c[0:2, 0:2] = np.dot(x.T, x)
        cr = np.linalg.cholesky(c)
        m = 1000 * k + 50 * k
        x = np.random.normal(size=(m, p))
        x = np.dot(x, cr.T)
        exog.append(x)
        endog.append(k * np.ones(m))
    endog = np.concatenate(endog)
    exog = np.concatenate(exog, axis=0)
    for dim in (1, 2, 3):
        cr = CORE(endog, exog, dim)
        pt = np.random.normal(size=(p, dim))
        (pt, _, _) = np.linalg.svd(pt, 0)
        gn = approx_fprime(pt.ravel(), cr.loglike, 1e-07)
        g = cr.score(pt.ravel())
        assert_allclose(g, gn, 1e-05, 1e-05)
        rslt = cr.fit()
        proj = rslt.params
        assert_equal(proj.shape[0], p)
        assert_equal(proj.shape[1], dim)
        assert_allclose(np.dot(proj.T, proj), np.eye(dim), 1e-08, 1e-08)
        if dim == 2:
            projt = np.zeros((p, 2))
            projt[0:2, 0:2] = np.eye(2)
            assert_allclose(np.trace(np.dot(proj.T, projt)), 2, rtol=0.001, atol=0.001)