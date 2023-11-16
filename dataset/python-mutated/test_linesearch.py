"""
Tests for line search routines
"""
from numpy.testing import assert_equal, assert_array_almost_equal, assert_array_almost_equal_nulp, assert_warns, suppress_warnings
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np

def assert_wolfe(s, phi, derphi, c1=0.0001, c2=0.9, err_msg=''):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check that strong Wolfe conditions apply\n    '
    phi1 = phi(s)
    phi0 = phi(0)
    derphi0 = derphi(0)
    derphi1 = derphi(s)
    msg = "s = {}; phi(0) = {}; phi(s) = {}; phi'(0) = {}; phi'(s) = {}; {}".format(s, phi0, phi1, derphi0, derphi1, err_msg)
    assert phi1 <= phi0 + c1 * s * derphi0, 'Wolfe 1 failed: ' + msg
    assert abs(derphi1) <= abs(c2 * derphi0), 'Wolfe 2 failed: ' + msg

def assert_armijo(s, phi, c1=0.0001, err_msg=''):
    if False:
        print('Hello World!')
    '\n    Check that Armijo condition applies\n    '
    phi1 = phi(s)
    phi0 = phi(0)
    msg = f's = {s}; phi(0) = {phi0}; phi(s) = {phi1}; {err_msg}'
    assert phi1 <= (1 - c1 * s) * phi0, msg

def assert_line_wolfe(x, p, s, f, fprime, **kw):
    if False:
        i = 10
        return i + 15
    assert_wolfe(s, phi=lambda sp: f(x + p * sp), derphi=lambda sp: np.dot(fprime(x + p * sp), p), **kw)

def assert_line_armijo(x, p, s, f, **kw):
    if False:
        return 10
    assert_armijo(s, phi=lambda sp: f(x + p * sp), **kw)

def assert_fp_equal(x, y, err_msg='', nulp=50):
    if False:
        print('Hello World!')
    'Assert two arrays are equal, up to some floating-point rounding error'
    try:
        assert_array_almost_equal_nulp(x, y, nulp)
    except AssertionError as e:
        raise AssertionError(f'{e}\n{err_msg}') from e

class TestLineSearch:

    def _scalar_func_1(self, s):
        if False:
            print('Hello World!')
        self.fcount += 1
        p = -s - s ** 3 + s ** 4
        dp = -1 - 3 * s ** 2 + 4 * s ** 3
        return (p, dp)

    def _scalar_func_2(self, s):
        if False:
            for i in range(10):
                print('nop')
        self.fcount += 1
        p = np.exp(-4 * s) + s ** 2
        dp = -4 * np.exp(-4 * s) + 2 * s
        return (p, dp)

    def _scalar_func_3(self, s):
        if False:
            return 10
        self.fcount += 1
        p = -np.sin(10 * s)
        dp = -10 * np.cos(10 * s)
        return (p, dp)

    def _line_func_1(self, x):
        if False:
            print('Hello World!')
        self.fcount += 1
        f = np.dot(x, x)
        df = 2 * x
        return (f, df)

    def _line_func_2(self, x):
        if False:
            return 10
        self.fcount += 1
        f = np.dot(x, np.dot(self.A, x)) + 1
        df = np.dot(self.A + self.A.T, x)
        return (f, df)

    def setup_method(self):
        if False:
            print('Hello World!')
        self.scalar_funcs = []
        self.line_funcs = []
        self.N = 20
        self.fcount = 0

        def bind_index(func, idx):
            if False:
                i = 10
                return i + 15
            return lambda *a, **kw: func(*a, **kw)[idx]
        for name in sorted(dir(self)):
            if name.startswith('_scalar_func_'):
                value = getattr(self, name)
                self.scalar_funcs.append((name, bind_index(value, 0), bind_index(value, 1)))
            elif name.startswith('_line_func_'):
                value = getattr(self, name)
                self.line_funcs.append((name, bind_index(value, 0), bind_index(value, 1)))
        np.random.seed(1234)
        self.A = np.random.randn(self.N, self.N)

    def scalar_iter(self):
        if False:
            print('Hello World!')
        for (name, phi, derphi) in self.scalar_funcs:
            for old_phi0 in np.random.randn(3):
                yield (name, phi, derphi, old_phi0)

    def line_iter(self):
        if False:
            return 10
        for (name, f, fprime) in self.line_funcs:
            k = 0
            while k < 9:
                x = np.random.randn(self.N)
                p = np.random.randn(self.N)
                if np.dot(p, fprime(x)) >= 0:
                    continue
                k += 1
                old_fv = float(np.random.randn())
                yield (name, f, fprime, x, p, old_fv)

    def test_scalar_search_wolfe1(self):
        if False:
            while True:
                i = 10
        c = 0
        for (name, phi, derphi, old_phi0) in self.scalar_iter():
            c += 1
            (s, phi1, phi0) = ls.scalar_search_wolfe1(phi, derphi, phi(0), old_phi0, derphi(0))
            assert_fp_equal(phi0, phi(0), name)
            assert_fp_equal(phi1, phi(s), name)
            assert_wolfe(s, phi, derphi, err_msg=name)
        assert c > 3

    def test_scalar_search_wolfe2(self):
        if False:
            i = 10
            return i + 15
        for (name, phi, derphi, old_phi0) in self.scalar_iter():
            (s, phi1, phi0, derphi1) = ls.scalar_search_wolfe2(phi, derphi, phi(0), old_phi0, derphi(0))
            assert_fp_equal(phi0, phi(0), name)
            assert_fp_equal(phi1, phi(s), name)
            if derphi1 is not None:
                assert_fp_equal(derphi1, derphi(s), name)
            assert_wolfe(s, phi, derphi, err_msg=f'{name} {old_phi0:g}')

    def test_scalar_search_wolfe2_with_low_amax(self):
        if False:
            return 10

        def phi(alpha):
            if False:
                i = 10
                return i + 15
            return (alpha - 5) ** 2

        def derphi(alpha):
            if False:
                i = 10
                return i + 15
            return 2 * (alpha - 5)
        (s, _, _, _) = assert_warns(LineSearchWarning, ls.scalar_search_wolfe2, phi, derphi, amax=0.001)
        assert s is None

    def test_scalar_search_wolfe2_regression(self):
        if False:
            while True:
                i = 10

        def phi(alpha):
            if False:
                return 10
            if alpha < 1:
                return -3 * np.pi / 2 * (alpha - 1)
            else:
                return np.cos(3 * np.pi / 2 * alpha - np.pi)

        def derphi(alpha):
            if False:
                return 10
            if alpha < 1:
                return -3 * np.pi / 2
            else:
                return -3 * np.pi / 2 * np.sin(3 * np.pi / 2 * alpha - np.pi)
        (s, _, _, _) = ls.scalar_search_wolfe2(phi, derphi)
        assert s < 1.5

    def test_scalar_search_armijo(self):
        if False:
            for i in range(10):
                print('nop')
        for (name, phi, derphi, old_phi0) in self.scalar_iter():
            (s, phi1) = ls.scalar_search_armijo(phi, phi(0), derphi(0))
            assert_fp_equal(phi1, phi(s), name)
            assert_armijo(s, phi, err_msg=f'{name} {old_phi0:g}')

    def test_line_search_wolfe1(self):
        if False:
            print('Hello World!')
        c = 0
        smax = 100
        for (name, f, fprime, x, p, old_f) in self.line_iter():
            f0 = f(x)
            g0 = fprime(x)
            self.fcount = 0
            (s, fc, gc, fv, ofv, gv) = ls.line_search_wolfe1(f, fprime, x, p, g0, f0, old_f, amax=smax)
            assert_equal(self.fcount, fc + gc)
            assert_fp_equal(ofv, f(x))
            if s is None:
                continue
            assert_fp_equal(fv, f(x + s * p))
            assert_array_almost_equal(gv, fprime(x + s * p), decimal=14)
            if s < smax:
                c += 1
                assert_line_wolfe(x, p, s, f, fprime, err_msg=name)
        assert c > 3

    def test_line_search_wolfe2(self):
        if False:
            return 10
        c = 0
        smax = 512
        for (name, f, fprime, x, p, old_f) in self.line_iter():
            f0 = f(x)
            g0 = fprime(x)
            self.fcount = 0
            with suppress_warnings() as sup:
                sup.filter(LineSearchWarning, 'The line search algorithm could not find a solution')
                sup.filter(LineSearchWarning, 'The line search algorithm did not converge')
                (s, fc, gc, fv, ofv, gv) = ls.line_search_wolfe2(f, fprime, x, p, g0, f0, old_f, amax=smax)
            assert_equal(self.fcount, fc + gc)
            assert_fp_equal(ofv, f(x))
            assert_fp_equal(fv, f(x + s * p))
            if gv is not None:
                assert_array_almost_equal(gv, fprime(x + s * p), decimal=14)
            if s < smax:
                c += 1
                assert_line_wolfe(x, p, s, f, fprime, err_msg=name)
        assert c > 3

    def test_line_search_wolfe2_bounds(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return np.dot(x, x)

        def fp(x):
            if False:
                return 10
            return 2 * x
        p = np.array([1, 0])
        x = -60 * p
        c2 = 0.5
        (s, _, _, _, _, _) = ls.line_search_wolfe2(f, fp, x, p, amax=30, c2=c2)
        assert_line_wolfe(x, p, s, f, fp)
        (s, _, _, _, _, _) = assert_warns(LineSearchWarning, ls.line_search_wolfe2, f, fp, x, p, amax=29, c2=c2)
        assert s is None
        assert_warns(LineSearchWarning, ls.line_search_wolfe2, f, fp, x, p, c2=c2, maxiter=5)

    def test_line_search_armijo(self):
        if False:
            while True:
                i = 10
        c = 0
        for (name, f, fprime, x, p, old_f) in self.line_iter():
            f0 = f(x)
            g0 = fprime(x)
            self.fcount = 0
            (s, fc, fv) = ls.line_search_armijo(f, x, p, g0, f0)
            c += 1
            assert_equal(self.fcount, fc)
            assert_fp_equal(fv, f(x + s * p))
            assert_line_armijo(x, p, s, f, err_msg=name)
        assert c >= 9

    def test_armijo_terminate_1(self):
        if False:
            print('Hello World!')
        count = [0]

        def phi(s):
            if False:
                while True:
                    i = 10
            count[0] += 1
            return -s + 0.01 * s ** 2
        (s, phi1) = ls.scalar_search_armijo(phi, phi(0), -1, alpha0=1)
        assert_equal(s, 1)
        assert_equal(count[0], 2)
        assert_armijo(s, phi)

    def test_wolfe_terminate(self):
        if False:
            return 10

        def phi(s):
            if False:
                while True:
                    i = 10
            count[0] += 1
            return -s + 0.05 * s ** 2

        def derphi(s):
            if False:
                print('Hello World!')
            count[0] += 1
            return -1 + 0.05 * 2 * s
        for func in [ls.scalar_search_wolfe1, ls.scalar_search_wolfe2]:
            count = [0]
            r = func(phi, derphi, phi(0), None, derphi(0))
            assert r[0] is not None, (r, func)
            assert count[0] <= 2 + 2, (count, func)
            assert_wolfe(r[0], phi, derphi, err_msg=str(func))