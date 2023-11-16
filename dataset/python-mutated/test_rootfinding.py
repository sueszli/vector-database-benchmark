"""

Created on Sat Mar 23 13:34:19 2013

Author: Josef Perktold
"""
import numpy as np
from statsmodels.tools.rootfinding import brentq_expanding
from numpy.testing import assert_allclose, assert_equal, assert_raises, assert_array_less

def func(x, a):
    if False:
        print('Hello World!')
    f = (x - a) ** 3
    return f

def func_nan(x, a, b):
    if False:
        for i in range(10):
            print('nop')
    x = np.atleast_1d(x)
    f = (x - 1.0 * a) ** 3
    f[x < b] = np.nan
    return f

def funcn(x, a):
    if False:
        while True:
            i = 10
    f = -(x - a) ** 3
    return f

def test_brentq_expanding():
    if False:
        i = 10
        return i + 15
    cases = [(0, {}), (50, {}), (-50, {}), (500000, dict(low=10000)), (-50000, dict(upp=-1000)), (500000, dict(low=300000, upp=700000)), (-50000, dict(low=-70000, upp=-1000))]
    funcs = [(func, None), (func, True), (funcn, None), (funcn, False)]
    for (f, inc) in funcs:
        for (a, kwds) in cases:
            kw = {'increasing': inc}
            kw.update(kwds)
            res = brentq_expanding(f, args=(a,), **kwds)
            assert_allclose(res, a, rtol=1e-05)
    assert_raises(ValueError, brentq_expanding, funcn, args=(-50000,), low=-40000, upp=-10000)
    assert_raises(ValueError, brentq_expanding, func, args=(-50000,), max_it=2)
    assert_raises(RuntimeError, brentq_expanding, func, args=(-50000,), maxiter_bq=3)
    assert_raises(ValueError, brentq_expanding, func_nan, args=(-20, 0.6))
    a = 500
    (val, info) = brentq_expanding(func, args=(a,), full_output=True)
    assert_allclose(val, a, rtol=1e-05)
    info1 = {'iterations': 63, 'start_bounds': (-1, 1), 'brentq_bounds': (100, 1000), 'flag': 'converged', 'function_calls': 64, 'iterations_expand': 3, 'converged': True}
    assert_array_less(info.iterations, 70)
    assert_array_less(info.function_calls, 70)
    for k in info1:
        if k in ['iterations', 'function_calls']:
            continue
        assert_equal(info1[k], getattr(info, k))
    assert_allclose(info.root, a, rtol=1e-05)