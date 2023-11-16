"""

Created on Mon Mar 18 15:48:23 2013
Author: Josef Perktold

TODO:
  - test behavior if nans or infs are encountered during the evaluation.
    now partially robust to nans, if increasing can be determined or is given.
  - rewrite core loop to use for...except instead of while.

"""
import numpy as np
from scipy import optimize
from statsmodels.tools.testing import Holder
DEBUG = False

def brentq_expanding(func, low=None, upp=None, args=(), xtol=1e-05, start_low=None, start_upp=None, increasing=None, max_it=100, maxiter_bq=100, factor=10, full_output=False):
    if False:
        return 10
    "find the root of a function in one variable by expanding and brentq\n\n    Assumes function ``func`` is monotonic.\n\n    Parameters\n    ----------\n    func : callable\n        function for which we find the root ``x`` such that ``func(x) = 0``\n    low : float or None\n        lower bound for brentq\n    upp : float or None\n        upper bound for brentq\n    args : tuple\n        optional additional arguments for ``func``\n    xtol : float\n        parameter x tolerance given to brentq\n    start_low : float (positive) or None\n        starting bound for expansion with increasing ``x``. It needs to be\n        positive. If None, then it is set to 1.\n    start_upp : float (negative) or None\n        starting bound for expansion with decreasing ``x``. It needs to be\n        negative. If None, then it is set to -1.\n    increasing : bool or None\n        If None, then the function is evaluated at the initial bounds to\n        determine wether the function is increasing or not. If increasing is\n        True (False), then it is assumed that the function is monotonically\n        increasing (decreasing).\n    max_it : int\n        maximum number of expansion steps.\n    maxiter_bq : int\n        maximum number of iterations of brentq.\n    factor : float\n        expansion factor for step of shifting the bounds interval, default is\n        10.\n    full_output : bool, optional\n        If full_output is False, the root is returned. If full_output is True,\n        the return value is (x, r), where x is the root, and r is a\n        RootResults object.\n\n\n    Returns\n    -------\n    x : float\n        root of the function, value at which ``func(x) = 0``.\n    info : RootResult (optional)\n        returned if ``full_output`` is True.\n        attributes:\n\n         - start_bounds : starting bounds for expansion stage\n         - brentq_bounds : bounds used with ``brentq``\n         - iterations_expand : number of iterations in expansion stage\n         - converged : True if brentq converged.\n         - flag : return status, 'converged' if brentq converged\n         - function_calls : number of function calls by ``brentq``\n         - iterations : number of iterations in ``brentq``\n\n\n    Notes\n    -----\n    If increasing is None, then whether the function is monotonically\n    increasing or decreasing is inferred from evaluating the function at the\n    initial bounds. This can fail if there is numerically no variation in the\n    data in this range. In this case, using different starting bounds or\n    directly specifying ``increasing`` can make it possible to move the\n    expansion in the right direction.\n\n    If\n\n    "
    (left, right) = (low, upp)
    if upp is not None:
        su = upp
    elif start_upp is not None:
        if start_upp < 0:
            raise ValueError('start_upp needs to be positive')
        su = start_upp
    else:
        su = 1.0
    if low is not None:
        sl = low
    elif start_low is not None:
        if start_low > 0:
            raise ValueError('start_low needs to be negative')
        sl = start_low
    else:
        sl = min(-1.0, su - 1.0)
    if upp is None:
        su = max(su, sl + 1.0)
    if (low is None or upp is None) and increasing is None:
        assert sl < su
        f_low = func(sl, *args)
        f_upp = func(su, *args)
        if np.max(np.abs(f_upp - f_low)) < 1e-15 and sl == -1 and (su == 1):
            sl = 1e-08
            f_low = func(sl, *args)
            increasing = f_low < f_upp
        delta = su - sl
        if np.isnan(f_low):
            for fraction in [0.25, 0.5, 0.75]:
                sl_ = sl + fraction * delta
                f_low = func(sl_, *args)
                if not np.isnan(f_low):
                    break
            else:
                raise ValueError('could not determine whether function is ' + 'increasing based on starting interval.' + '\nspecify increasing or change starting ' + 'bounds')
        if np.isnan(f_upp):
            for fraction in [0.25, 0.5, 0.75]:
                su_ = su + fraction * delta
                f_upp = func(su_, *args)
                if not np.isnan(f_upp):
                    break
            else:
                raise ValueError('could not determine whether function is' + 'increasing based on starting interval.' + '\nspecify increasing or change starting ' + 'bounds')
        increasing = f_low < f_upp
    if not increasing:
        (sl, su) = (su, sl)
        (left, right) = (right, left)
    n_it = 0
    if left is None and sl != 0:
        left = sl
        while func(left, *args) > 0:
            right = left
            left *= factor
            if n_it >= max_it:
                break
            n_it += 1
    if right is None and su != 0:
        right = su
        while func(right, *args) < 0:
            left = right
            right *= factor
            if n_it >= max_it:
                break
            n_it += 1
    if n_it >= max_it:
        f_low = func(sl, *args)
        f_upp = func(su, *args)
        if np.isnan(f_low) and np.isnan(f_upp):
            raise ValueError('max_it reached' + '\nthe function values at boths bounds are NaN' + '\nchange the starting bounds, set bounds' + 'or increase max_it')
    res = optimize.brentq(func, left, right, args=args, xtol=xtol, maxiter=maxiter_bq, full_output=full_output)
    if full_output:
        val = res[0]
        info = Holder(root=res[1].root, iterations=res[1].iterations, function_calls=res[1].function_calls, converged=res[1].converged, flag=res[1].flag, iterations_expand=n_it, start_bounds=(sl, su), brentq_bounds=(left, right), increasing=increasing)
        return (val, info)
    else:
        return res