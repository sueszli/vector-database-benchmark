"""

Created on Sat Mar 23 13:35:51 2013

Author: Josef Perktold
"""
import numpy as np
from statsmodels.tools.rootfinding import brentq_expanding
DEBUG = False

def func(x, a):
    if False:
        for i in range(10):
            print('nop')
    f = (x - a) ** 3
    if DEBUG:
        print('evaluating at %g, fval = %f' % (x, f))
    return f

def func_nan(x, a, b):
    if False:
        while True:
            i = 10
    x = np.atleast_1d(x)
    f = (x - 1.0 * a) ** 3
    f[x < b] = np.nan
    if DEBUG:
        print('evaluating at %f, fval = %f' % (x, f))
    return f

def funcn(x, a):
    if False:
        i = 10
        return i + 15
    f = -(x - a) ** 3
    if DEBUG:
        print('evaluating at %g, fval = %g' % (x, f))
    return f

def func2(x, a):
    if False:
        while True:
            i = 10
    f = (x - a) ** 3
    print('evaluating at %g, fval = %f' % (x, f))
    return f
if __name__ == '__main__':
    run_all = False
    if run_all:
        print(brentq_expanding(func, args=(0,), increasing=True))
        print(brentq_expanding(funcn, args=(0,), increasing=False))
        print(brentq_expanding(funcn, args=(-50,), increasing=False))
        print(brentq_expanding(func, args=(20,)))
        print(brentq_expanding(funcn, args=(20,)))
        print(brentq_expanding(func, args=(500000,)))
        print(brentq_expanding(func, args=(500000,), low=10000))
        print(brentq_expanding(func, args=(-50000,), upp=-1000))
        print(brentq_expanding(funcn, args=(500000,), low=10000))
        print(brentq_expanding(funcn, args=(-50000,), upp=-1000))
        print(brentq_expanding(func, args=(500000,), low=300000, upp=700000))
        print(brentq_expanding(func, args=(-50000,), low=-70000, upp=-1000))
        print(brentq_expanding(funcn, args=(500000,), low=300000, upp=700000))
        print(brentq_expanding(funcn, args=(-50000,), low=-70000, upp=-10000))
        print(brentq_expanding(func, args=(1.234e+30,), xtol=10000000000.0, increasing=True, maxiter_bq=200))
    print(brentq_expanding(func, args=(-50000,), start_low=-10000))
    try:
        print(brentq_expanding(func, args=(-500,), start_upp=-100))
    except ValueError:
        print('raised ValueError start_upp needs to be positive')
    " it still works\n    raise ValueError('start_upp needs to be positive')\n    -499.999996336\n    "
    " this does not work\n    >>> print(brentq_expanding(func, args=(-500,), start_upp=-1000)\n    raise ValueError('start_upp needs to be positive')\n    OverflowError: (34, 'Result too large')\n    "
    try:
        print(brentq_expanding(funcn, args=(-50000,), low=-40000, upp=-10000))
    except Exception as e:
        print(e)
    (val, info) = brentq_expanding(func, args=(500,), full_output=True)
    print(val)
    print(vars(info))
    print(brentq_expanding(func_nan, args=(20, 0), increasing=True))
    print(brentq_expanding(func_nan, args=(20, 0)))
    print(brentq_expanding(func_nan, args=(-20, 0), increasing=True))
    print(brentq_expanding(func_nan, args=(-20, 0)))