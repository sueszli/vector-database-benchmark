import itertools
import numpy as np
from numpy import exp
from numpy.testing import assert_, assert_equal
from scipy.optimize import root

def test_performance():
    if False:
        print('Hello World!')
    e_a = 1e-05
    e_r = 0.0001
    table_1 = [dict(F=F_1, x0=x0_1, n=1000, nit=5, nfev=5), dict(F=F_1, x0=x0_1, n=10000, nit=2, nfev=2), dict(F=F_2, x0=x0_2, n=500, nit=11, nfev=11), dict(F=F_2, x0=x0_2, n=2000, nit=11, nfev=11), dict(F=F_6, x0=x0_6, n=100, nit=6, nfev=6), dict(F=F_7, x0=x0_7, n=99, nit=23, nfev=29), dict(F=F_7, x0=x0_7, n=999, nit=23, nfev=29), dict(F=F_9, x0=x0_9, n=100, nit=12, nfev=18), dict(F=F_9, x0=x0_9, n=1000, nit=12, nfev=18), dict(F=F_10, x0=x0_10, n=1000, nit=5, nfev=5)]
    for (xscale, yscale, line_search) in itertools.product([1.0, 1e-10, 10000000000.0], [1.0, 1e-10, 10000000000.0], ['cruz', 'cheng']):
        for problem in table_1:
            n = problem['n']

            def func(x, n):
                if False:
                    i = 10
                    return i + 15
                return yscale * problem['F'](x / xscale, n)
            args = (n,)
            x0 = problem['x0'](n) * xscale
            fatol = np.sqrt(n) * e_a * yscale + e_r * np.linalg.norm(func(x0, n))
            sigma_eps = 1e-10 * min(yscale / xscale, xscale / yscale)
            sigma_0 = xscale / yscale
            with np.errstate(over='ignore'):
                sol = root(func, x0, args=args, options=dict(ftol=0, fatol=fatol, maxfev=problem['nfev'] + 1, sigma_0=sigma_0, sigma_eps=sigma_eps, line_search=line_search), method='DF-SANE')
            err_msg = repr([xscale, yscale, line_search, problem, np.linalg.norm(func(sol.x, n)), fatol, sol.success, sol.nit, sol.nfev])
            assert_(sol.success, err_msg)
            assert_(sol.nfev <= problem['nfev'] + 1, err_msg)
            assert_(sol.nit <= problem['nit'], err_msg)
            assert_(np.linalg.norm(func(sol.x, n)) <= fatol, err_msg)

def test_complex():
    if False:
        return 10

    def func(z):
        if False:
            i = 10
            return i + 15
        return z ** 2 - 1 + 2j
    x0 = 2j
    ftol = 0.0001
    sol = root(func, x0, tol=ftol, method='DF-SANE')
    assert_(sol.success)
    f0 = np.linalg.norm(func(x0))
    fx = np.linalg.norm(func(sol.x))
    assert_(fx <= ftol * f0)

def test_linear_definite():
    if False:
        return 10

    def check_solvability(A, b, line_search='cruz'):
        if False:
            return 10

        def func(x):
            if False:
                print('Hello World!')
            return A.dot(x) - b
        xp = np.linalg.solve(A, b)
        eps = np.linalg.norm(func(xp)) * 1000.0
        sol = root(func, b, options=dict(fatol=eps, ftol=0, maxfev=17523, line_search=line_search), method='DF-SANE')
        assert_(sol.success)
        assert_(np.linalg.norm(func(sol.x)) <= eps)
    n = 90
    np.random.seed(1234)
    A = np.arange(n * n).reshape(n, n)
    A = A + n * n * np.diag(1 + np.arange(n))
    assert_(np.linalg.eigvals(A).min() > 0)
    b = np.arange(n) * 1.0
    check_solvability(A, b, 'cruz')
    check_solvability(A, b, 'cheng')
    check_solvability(-A, b, 'cruz')
    check_solvability(-A, b, 'cheng')

def test_shape():
    if False:
        print('Hello World!')

    def f(x, arg):
        if False:
            for i in range(10):
                print('nop')
        return x - arg
    for dt in [float, complex]:
        x = np.zeros([2, 2])
        arg = np.ones([2, 2], dtype=dt)
        sol = root(f, x, args=(arg,), method='DF-SANE')
        assert_(sol.success)
        assert_equal(sol.x.shape, x.shape)

def F_1(x, n):
    if False:
        i = 10
        return i + 15
    g = np.zeros([n])
    i = np.arange(2, n + 1)
    g[0] = exp(x[0] - 1) - 1
    g[1:] = i * (exp(x[1:] - 1) - x[1:])
    return g

def x0_1(n):
    if False:
        return 10
    x0 = np.empty([n])
    x0.fill(n / (n - 1))
    return x0

def F_2(x, n):
    if False:
        for i in range(10):
            print('nop')
    g = np.zeros([n])
    i = np.arange(2, n + 1)
    g[0] = exp(x[0]) - 1
    g[1:] = 0.1 * i * (exp(x[1:]) + x[:-1] - 1)
    return g

def x0_2(n):
    if False:
        return 10
    x0 = np.empty([n])
    x0.fill(1 / n ** 2)
    return x0

def F_4(x, n):
    if False:
        while True:
            i = 10
    assert_equal(n % 3, 0)
    g = np.zeros([n])
    g[::3] = 0.6 * x[::3] + 1.6 * x[1::3] ** 3 - 7.2 * x[1::3] ** 2 + 9.6 * x[1::3] - 4.8
    g[1::3] = 0.48 * x[::3] - 0.72 * x[1::3] ** 3 + 3.24 * x[1::3] ** 2 - 4.32 * x[1::3] - x[2::3] + 0.2 * x[2::3] ** 3 + 2.16
    g[2::3] = 1.25 * x[2::3] - 0.25 * x[2::3] ** 3
    return g

def x0_4(n):
    if False:
        return 10
    assert_equal(n % 3, 0)
    x0 = np.array([-1, 1 / 2, -1] * (n // 3))
    return x0

def F_6(x, n):
    if False:
        i = 10
        return i + 15
    c = 0.9
    mu = (np.arange(1, n + 1) - 0.5) / n
    return x - 1 / (1 - c / (2 * n) * (mu[:, None] * x / (mu[:, None] + mu)).sum(axis=1))

def x0_6(n):
    if False:
        print('Hello World!')
    return np.ones([n])

def F_7(x, n):
    if False:
        i = 10
        return i + 15
    assert_equal(n % 3, 0)

    def phi(t):
        if False:
            print('Hello World!')
        v = 0.5 * t - 2
        v[t > -1] = ((-592 * t ** 3 + 888 * t ** 2 + 4551 * t - 1924) / 1998)[t > -1]
        v[t >= 2] = (0.5 * t + 2)[t >= 2]
        return v
    g = np.zeros([n])
    g[::3] = 10000.0 * x[1::3] ** 2 - 1
    g[1::3] = exp(-x[::3]) + exp(-x[1::3]) - 1.0001
    g[2::3] = phi(x[2::3])
    return g

def x0_7(n):
    if False:
        i = 10
        return i + 15
    assert_equal(n % 3, 0)
    return np.array([0.001, 18, 1] * (n // 3))

def F_9(x, n):
    if False:
        while True:
            i = 10
    g = np.zeros([n])
    i = np.arange(2, n)
    g[0] = x[0] ** 3 / 3 + x[1] ** 2 / 2
    g[1:-1] = -x[1:-1] ** 2 / 2 + i * x[1:-1] ** 3 / 3 + x[2:] ** 2 / 2
    g[-1] = -x[-1] ** 2 / 2 + n * x[-1] ** 3 / 3
    return g

def x0_9(n):
    if False:
        for i in range(10):
            print('nop')
    return np.ones([n])

def F_10(x, n):
    if False:
        while True:
            i = 10
    return np.log(1 + x) - x / n

def x0_10(n):
    if False:
        print('Hello World!')
    return np.ones([n])