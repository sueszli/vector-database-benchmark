"""Compute a grid of values for Wright's generalized Bessel function
and save the values to data files for use in tests. Using mpmath directly in
tests would take too long.

This takes about 10 minutes to run on a 2.7 GHz i7 Macbook Pro.
"""
from functools import lru_cache
import os
from time import time
import numpy as np
from scipy.special._mptestutils import mpf2float
try:
    import mpmath as mp
except ImportError:
    pass
exp_inf = 709.7827128933841

@lru_cache(maxsize=100000)
def rgamma_cached(x, dps):
    if False:
        return 10
    with mp.workdps(dps):
        return mp.rgamma(x)

def mp_wright_bessel(a, b, x, dps=50, maxterms=2000):
    if False:
        i = 10
        return i + 15
    "Compute Wright's generalized Bessel function as Series with mpmath.\n    "
    with mp.workdps(dps):
        (a, b, x) = (mp.mpf(a), mp.mpf(b), mp.mpf(x))
        res = mp.nsum(lambda k: x ** k / mp.fac(k) * rgamma_cached(a * k + b, dps=dps), [0, mp.inf], tol=dps, method='s', steps=[maxterms])
        return mpf2float(res)

def main():
    if False:
        i = 10
        return i + 15
    t0 = time()
    print(__doc__)
    pwd = os.path.dirname(__file__)
    eps = np.finfo(float).eps * 100
    a_range = np.array([eps, 0.0001 * (1 - eps), 0.0001, 0.0001 * (1 + eps), 0.001 * (1 - eps), 0.001, 0.001 * (1 + eps), 0.1, 0.5, 1 * (1 - eps), 1, 1 * (1 + eps), 1.5, 2, 4.999, 5, 10])
    b_range = np.array([0, eps, 1e-10, 1e-05, 0.1, 1, 2, 10, 20, 100])
    x_range = np.array([0, eps, 1 - eps, 1, 1 + eps, 1.5, 2 - eps, 2, 2 + eps, 9 - eps, 9, 9 + eps, 10 * (1 - eps), 10, 10 * (1 + eps), 100 * (1 - eps), 100, 100 * (1 + eps), 500, exp_inf, 1000.0, 100000.0, 10000000000.0, 1e+20])
    (a_range, b_range, x_range) = np.meshgrid(a_range, b_range, x_range, indexing='ij')
    a_range = a_range.flatten()
    b_range = b_range.flatten()
    x_range = x_range.flatten()
    bool_filter = ~((a_range < 0.005) & (x_range >= exp_inf))
    bool_filter = bool_filter & ~((a_range < 0.2) & (x_range > exp_inf))
    bool_filter = bool_filter & ~((a_range < 0.5) & (x_range > 1000.0))
    bool_filter = bool_filter & ~((a_range < 0.56) & (x_range > 5000.0))
    bool_filter = bool_filter & ~((a_range < 1) & (x_range > 10000.0))
    bool_filter = bool_filter & ~((a_range < 1.4) & (x_range > 100000.0))
    bool_filter = bool_filter & ~((a_range < 1.8) & (x_range > 1000000.0))
    bool_filter = bool_filter & ~((a_range < 2.2) & (x_range > 10000000.0))
    bool_filter = bool_filter & ~((a_range < 2.5) & (x_range > 100000000.0))
    bool_filter = bool_filter & ~((a_range < 2.9) & (x_range > 1000000000.0))
    bool_filter = bool_filter & ~((a_range < 3.3) & (x_range > 10000000000.0))
    bool_filter = bool_filter & ~((a_range < 3.7) & (x_range > 100000000000.0))
    bool_filter = bool_filter & ~((a_range < 4) & (x_range > 1000000000000.0))
    bool_filter = bool_filter & ~((a_range < 4.4) & (x_range > 10000000000000.0))
    bool_filter = bool_filter & ~((a_range < 4.7) & (x_range > 100000000000000.0))
    bool_filter = bool_filter & ~((a_range < 5.1) & (x_range > 1000000000000000.0))
    bool_filter = bool_filter & ~((a_range < 5.4) & (x_range > 1e+16))
    bool_filter = bool_filter & ~((a_range < 5.8) & (x_range > 1e+17))
    bool_filter = bool_filter & ~((a_range < 6.2) & (x_range > 1e+18))
    bool_filter = bool_filter & ~((a_range < 6.2) & (x_range > 1e+18))
    bool_filter = bool_filter & ~((a_range < 6.5) & (x_range > 1e+19))
    bool_filter = bool_filter & ~((a_range < 6.9) & (x_range > 1e+20))
    failing = np.array([[0.1, 100, 709.7827128933841], [0.5, 10, 709.7827128933841], [0.5, 10, 1000], [0.5, 100, 1000], [1, 20, 100000], [1, 100, 100000], [1.0000000000000222, 20, 100000], [1.0000000000000222, 100, 100000], [1.5, 0, 500], [1.5, 2.220446049250313e-14, 500], [1.5, 1e-10, 500], [1.5, 1e-05, 500], [1.5, 0.1, 500], [1.5, 20, 100000], [1.5, 100, 100000]]).tolist()
    does_fail = np.full_like(a_range, False, dtype=bool)
    for i in range(x_range.size):
        if [a_range[i], b_range[i], x_range[i]] in failing:
            does_fail[i] = True
    a_range = a_range[bool_filter]
    b_range = b_range[bool_filter]
    x_range = x_range[bool_filter]
    does_fail = does_fail[bool_filter]
    dataset = []
    print(f'Computing {x_range.size} single points.')
    print('Tests will fail for the following data points:')
    for i in range(x_range.size):
        a = a_range[i]
        b = b_range[i]
        x = x_range[i]
        maxterms = 1000
        if a < 1e-06 and x >= exp_inf / 10:
            maxterms = 2000
        f = mp_wright_bessel(a, b, x, maxterms=maxterms)
        if does_fail[i]:
            print(f'failing data point a, b, x, value = [{a}, {b}, {x}, {f}]')
        else:
            dataset.append((a, b, x, f))
    dataset = np.array(dataset)
    filename = os.path.join(pwd, '..', 'tests', 'data', 'local', 'wright_bessel.txt')
    np.savetxt(filename, dataset)
    print(f'{(time() - t0) / 60:.1f} minutes elapsed')
if __name__ == '__main__':
    main()