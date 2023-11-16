import sys
import platform
import pytest
import numpy as np
import numpy._core._multiarray_umath as ncu
from numpy.testing import assert_raises, assert_equal, assert_array_equal, assert_almost_equal, assert_array_max_ulp
with np.errstate(all='ignore'):
    functions_seem_flaky = np.exp(complex(np.inf, 0)).imag != 0 or np.log(complex(ncu.NZERO, 0)).imag != np.pi
xfail_complex_tests = not sys.platform.startswith('linux') or functions_seem_flaky
platform_skip = pytest.mark.skipif(xfail_complex_tests, reason='Inadequate C99 complex support')

class TestCexp:

    def test_simple(self):
        if False:
            while True:
                i = 10
        check = check_complex_value
        f = np.exp
        check(f, 1, 0, np.exp(1), 0, False)
        check(f, 0, 1, np.cos(1), np.sin(1), False)
        ref = np.exp(1) * complex(np.cos(1), np.sin(1))
        check(f, 1, 1, ref.real, ref.imag, False)

    @platform_skip
    def test_special_values(self):
        if False:
            print('Hello World!')
        check = check_complex_value
        f = np.exp
        check(f, ncu.PZERO, 0, 1, 0, False)
        check(f, ncu.NZERO, 0, 1, 0, False)
        check(f, 1, np.inf, np.nan, np.nan)
        check(f, -1, np.inf, np.nan, np.nan)
        check(f, 0, np.inf, np.nan, np.nan)
        check(f, np.inf, 0, np.inf, 0)
        check(f, -np.inf, 1, ncu.PZERO, ncu.PZERO)
        check(f, -np.inf, 0.75 * np.pi, ncu.NZERO, ncu.PZERO)
        check(f, np.inf, 1, np.inf, np.inf)
        check(f, np.inf, 0.75 * np.pi, -np.inf, np.inf)

        def _check_ninf_inf(dummy):
            if False:
                print('Hello World!')
            msgform = 'cexp(-inf, inf) is (%f, %f), expected (+-0, +-0)'
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(-np.inf, np.inf)))
                if z.real != 0 or z.imag != 0:
                    raise AssertionError(msgform % (z.real, z.imag))
        _check_ninf_inf(None)

        def _check_inf_inf(dummy):
            if False:
                while True:
                    i = 10
            msgform = 'cexp(inf, inf) is (%f, %f), expected (+-inf, nan)'
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(np.inf, np.inf)))
                if not np.isinf(z.real) or not np.isnan(z.imag):
                    raise AssertionError(msgform % (z.real, z.imag))
        _check_inf_inf(None)

        def _check_ninf_nan(dummy):
            if False:
                while True:
                    i = 10
            msgform = 'cexp(-inf, nan) is (%f, %f), expected (+-0, +-0)'
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(-np.inf, np.nan)))
                if z.real != 0 or z.imag != 0:
                    raise AssertionError(msgform % (z.real, z.imag))
        _check_ninf_nan(None)

        def _check_inf_nan(dummy):
            if False:
                print('Hello World!')
            msgform = 'cexp(-inf, nan) is (%f, %f), expected (+-inf, nan)'
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(np.inf, np.nan)))
                if not np.isinf(z.real) or not np.isnan(z.imag):
                    raise AssertionError(msgform % (z.real, z.imag))
        _check_inf_nan(None)
        check(f, np.nan, 1, np.nan, np.nan)
        check(f, np.nan, -1, np.nan, np.nan)
        check(f, np.nan, np.inf, np.nan, np.nan)
        check(f, np.nan, -np.inf, np.nan, np.nan)
        check(f, np.nan, np.nan, np.nan, np.nan)

    @pytest.mark.skip(reason='cexp(nan + 0I) is wrong on most platforms')
    def test_special_values2(self):
        if False:
            print('Hello World!')
        check = check_complex_value
        f = np.exp
        check(f, np.nan, 0, np.nan, 0)

class TestClog:

    def test_simple(self):
        if False:
            print('Hello World!')
        x = np.array([1 + 0j, 1 + 2j])
        y_r = np.log(np.abs(x)) + 1j * np.angle(x)
        y = np.log(x)
        assert_almost_equal(y, y_r)

    @platform_skip
    @pytest.mark.skipif(platform.machine() == 'armv5tel', reason='See gh-413.')
    def test_special_values(self):
        if False:
            return 10
        xl = []
        yl = []
        with np.errstate(divide='raise'):
            x = np.array([ncu.NZERO], dtype=complex)
            y = complex(-np.inf, np.pi)
            assert_raises(FloatingPointError, np.log, x)
        with np.errstate(divide='ignore'):
            assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        with np.errstate(divide='raise'):
            x = np.array([0], dtype=complex)
            y = complex(-np.inf, 0)
            assert_raises(FloatingPointError, np.log, x)
        with np.errstate(divide='ignore'):
            assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(1, np.inf)], dtype=complex)
        y = complex(np.inf, 0.5 * np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(-1, np.inf)], dtype=complex)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        with np.errstate(invalid='raise'):
            x = np.array([complex(1.0, np.nan)], dtype=complex)
            y = complex(np.nan, np.nan)
        with np.errstate(invalid='ignore'):
            assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        with np.errstate(invalid='raise'):
            x = np.array([np.inf + 1j * np.nan], dtype=complex)
        with np.errstate(invalid='ignore'):
            assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([-np.inf + 1j], dtype=complex)
        y = complex(np.inf, np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([np.inf + 1j], dtype=complex)
        y = complex(np.inf, 0)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(-np.inf, np.inf)], dtype=complex)
        y = complex(np.inf, 0.75 * np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(np.inf, np.inf)], dtype=complex)
        y = complex(np.inf, 0.25 * np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(np.inf, np.nan)], dtype=complex)
        y = complex(np.inf, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(-np.inf, np.nan)], dtype=complex)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(np.nan, 1)], dtype=complex)
        y = complex(np.nan, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(np.nan, np.inf)], dtype=complex)
        y = complex(np.inf, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        x = np.array([complex(np.nan, np.nan)], dtype=complex)
        y = complex(np.nan, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)
        xa = np.array(xl, dtype=complex)
        ya = np.array(yl, dtype=complex)
        with np.errstate(divide='ignore'):
            for i in range(len(xa)):
                assert_almost_equal(np.log(xa[i].conj()), ya[i].conj())

class TestCsqrt:

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        check_complex_value(np.sqrt, 1, 0, 1, 0)
        rres = 0.5 * np.sqrt(2)
        ires = rres
        check_complex_value(np.sqrt, 0, 1, rres, ires, False)
        check_complex_value(np.sqrt, -1, 0, 0, 1)

    def test_simple_conjugate(self):
        if False:
            while True:
                i = 10
        ref = np.conj(np.sqrt(complex(1, 1)))

        def f(z):
            if False:
                for i in range(10):
                    print('nop')
            return np.sqrt(np.conj(z))
        check_complex_value(f, 1, 1, ref.real, ref.imag, False)

    @platform_skip
    def test_special_values(self):
        if False:
            for i in range(10):
                print('nop')
        check = check_complex_value
        f = np.sqrt
        check(f, ncu.PZERO, 0, 0, 0)
        check(f, ncu.NZERO, 0, 0, 0)
        check(f, 1, np.inf, np.inf, np.inf)
        check(f, -1, np.inf, np.inf, np.inf)
        check(f, ncu.PZERO, np.inf, np.inf, np.inf)
        check(f, ncu.NZERO, np.inf, np.inf, np.inf)
        check(f, np.inf, np.inf, np.inf, np.inf)
        check(f, -np.inf, np.inf, np.inf, np.inf)
        check(f, -np.nan, np.inf, np.inf, np.inf)
        check(f, 1, np.nan, np.nan, np.nan)
        check(f, -1, np.nan, np.nan, np.nan)
        check(f, 0, np.nan, np.nan, np.nan)
        check(f, -np.inf, 1, ncu.PZERO, np.inf)
        check(f, np.inf, 1, np.inf, ncu.PZERO)

        def _check_ninf_nan(dummy):
            if False:
                print('Hello World!')
            msgform = 'csqrt(-inf, nan) is (%f, %f), expected (nan, +-inf)'
            z = np.sqrt(np.array(complex(-np.inf, np.nan)))
            with np.errstate(invalid='ignore'):
                if not (np.isnan(z.real) and np.isinf(z.imag)):
                    raise AssertionError(msgform % (z.real, z.imag))
        _check_ninf_nan(None)
        check(f, np.inf, np.nan, np.inf, np.nan)
        check(f, np.nan, 0, np.nan, np.nan)
        check(f, np.nan, 1, np.nan, np.nan)
        check(f, np.nan, np.nan, np.nan, np.nan)

class TestCpow:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.olderr = np.seterr(invalid='ignore')

    def teardown_method(self):
        if False:
            for i in range(10):
                print('nop')
        np.seterr(**self.olderr)

    def test_simple(self):
        if False:
            while True:
                i = 10
        x = np.array([1 + 1j, 0 + 2j, 1 + 2j, np.inf, np.nan])
        y_r = x ** 2
        y = np.power(x, 2)
        assert_almost_equal(y, y_r)

    def test_scalar(self):
        if False:
            while True:
                i = 10
        x = np.array([1, 1j, 2, 2.5 + 0.37j, np.inf, np.nan])
        y = np.array([1, 1j, -0.5 + 1.5j, -0.5 + 1.5j, 2, 3])
        lx = list(range(len(x)))
        p_r = [1 + 0j, 0.20787957635076193 + 0j, 0.35812203996480685 + 0.6097119028618724j, 0.12659112128185032 + 0.48847676699581527j, complex(np.inf, np.nan), complex(np.nan, np.nan)]
        n_r = [x[i] ** y[i] for i in lx]
        for i in lx:
            assert_almost_equal(n_r[i], p_r[i], err_msg='Loop %d\n' % i)

    def test_array(self):
        if False:
            i = 10
            return i + 15
        x = np.array([1, 1j, 2, 2.5 + 0.37j, np.inf, np.nan])
        y = np.array([1, 1j, -0.5 + 1.5j, -0.5 + 1.5j, 2, 3])
        lx = list(range(len(x)))
        p_r = [1 + 0j, 0.20787957635076193 + 0j, 0.35812203996480685 + 0.6097119028618724j, 0.12659112128185032 + 0.48847676699581527j, complex(np.inf, np.nan), complex(np.nan, np.nan)]
        n_r = x ** y
        for i in lx:
            assert_almost_equal(n_r[i], p_r[i], err_msg='Loop %d\n' % i)

class TestCabs:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.olderr = np.seterr(invalid='ignore')

    def teardown_method(self):
        if False:
            for i in range(10):
                print('nop')
        np.seterr(**self.olderr)

    def test_simple(self):
        if False:
            print('Hello World!')
        x = np.array([1 + 1j, 0 + 2j, 1 + 2j, np.inf, np.nan])
        y_r = np.array([np.sqrt(2.0), 2, np.sqrt(5), np.inf, np.nan])
        y = np.abs(x)
        assert_almost_equal(y, y_r)

    def test_fabs(self):
        if False:
            i = 10
            return i + 15
        x = np.array([1 + 0j], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))
        x = np.array([complex(1, ncu.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))
        x = np.array([complex(np.inf, ncu.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))
        x = np.array([complex(np.nan, ncu.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

    def test_cabs_inf_nan(self):
        if False:
            print('Hello World!')
        (x, y) = ([], [])
        x.append(np.nan)
        y.append(np.nan)
        check_real_value(np.abs, np.nan, np.nan, np.nan)
        x.append(np.nan)
        y.append(-np.nan)
        check_real_value(np.abs, -np.nan, np.nan, np.nan)
        x.append(np.inf)
        y.append(np.nan)
        check_real_value(np.abs, np.inf, np.nan, np.inf)
        x.append(-np.inf)
        y.append(np.nan)
        check_real_value(np.abs, -np.inf, np.nan, np.inf)

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            return np.abs(np.conj(a))

        def g(a, b):
            if False:
                while True:
                    i = 10
            return np.abs(complex(a, b))
        xa = np.array(x, dtype=complex)
        assert len(xa) == len(x) == len(y)
        for (xi, yi) in zip(x, y):
            ref = g(xi, yi)
            check_real_value(f, xi, yi, ref)

class TestCarg:

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        check_real_value(ncu._arg, 1, 0, 0, False)
        check_real_value(ncu._arg, 0, 1, 0.5 * np.pi, False)
        check_real_value(ncu._arg, 1, 1, 0.25 * np.pi, False)
        check_real_value(ncu._arg, ncu.PZERO, ncu.PZERO, ncu.PZERO)

    @pytest.mark.skip(reason='Complex arithmetic with signed zero fails on most platforms')
    def test_zero(self):
        if False:
            return 10
        check_real_value(ncu._arg, ncu.NZERO, ncu.PZERO, np.pi, False)
        check_real_value(ncu._arg, ncu.NZERO, ncu.NZERO, -np.pi, False)
        check_real_value(ncu._arg, ncu.PZERO, ncu.PZERO, ncu.PZERO)
        check_real_value(ncu._arg, ncu.PZERO, ncu.NZERO, ncu.NZERO)
        check_real_value(ncu._arg, 1, ncu.PZERO, ncu.PZERO, False)
        check_real_value(ncu._arg, 1, ncu.NZERO, ncu.NZERO, False)
        check_real_value(ncu._arg, -1, ncu.PZERO, np.pi, False)
        check_real_value(ncu._arg, -1, ncu.NZERO, -np.pi, False)
        check_real_value(ncu._arg, ncu.PZERO, 1, 0.5 * np.pi, False)
        check_real_value(ncu._arg, ncu.NZERO, 1, 0.5 * np.pi, False)
        check_real_value(ncu._arg, ncu.PZERO, -1, 0.5 * np.pi, False)
        check_real_value(ncu._arg, ncu.NZERO, -1, -0.5 * np.pi, False)

    def test_special_values(self):
        if False:
            for i in range(10):
                print('nop')
        check_real_value(ncu._arg, -np.inf, 1, np.pi, False)
        check_real_value(ncu._arg, -np.inf, -1, -np.pi, False)
        check_real_value(ncu._arg, np.inf, 1, ncu.PZERO, False)
        check_real_value(ncu._arg, np.inf, -1, ncu.NZERO, False)
        check_real_value(ncu._arg, 1, np.inf, 0.5 * np.pi, False)
        check_real_value(ncu._arg, 1, -np.inf, -0.5 * np.pi, False)
        check_real_value(ncu._arg, -np.inf, np.inf, 0.75 * np.pi, False)
        check_real_value(ncu._arg, -np.inf, -np.inf, -0.75 * np.pi, False)
        check_real_value(ncu._arg, np.inf, np.inf, 0.25 * np.pi, False)
        check_real_value(ncu._arg, np.inf, -np.inf, -0.25 * np.pi, False)
        check_real_value(ncu._arg, np.nan, 0, np.nan, False)
        check_real_value(ncu._arg, 0, np.nan, np.nan, False)
        check_real_value(ncu._arg, np.nan, np.inf, np.nan, False)
        check_real_value(ncu._arg, np.inf, np.nan, np.nan, False)

def check_real_value(f, x1, y1, x, exact=True):
    if False:
        print('Hello World!')
    z1 = np.array([complex(x1, y1)])
    if exact:
        assert_equal(f(z1), x)
    else:
        assert_almost_equal(f(z1), x)

def check_complex_value(f, x1, y1, x2, y2, exact=True):
    if False:
        i = 10
        return i + 15
    z1 = np.array([complex(x1, y1)])
    z2 = complex(x2, y2)
    with np.errstate(invalid='ignore'):
        if exact:
            assert_equal(f(z1), z2)
        else:
            assert_almost_equal(f(z1), z2)

class TestSpecialComplexAVX:

    @pytest.mark.parametrize('stride', [-4, -2, -1, 1, 2, 4])
    @pytest.mark.parametrize('astype', [np.complex64, np.complex128])
    def test_array(self, stride, astype):
        if False:
            return 10
        arr = np.array([complex(np.nan, np.nan), complex(np.nan, np.inf), complex(np.inf, np.nan), complex(np.inf, np.inf), complex(0.0, np.inf), complex(np.inf, 0.0), complex(0.0, 0.0), complex(0.0, np.nan), complex(np.nan, 0.0)], dtype=astype)
        abs_true = np.array([np.nan, np.inf, np.inf, np.inf, np.inf, np.inf, 0.0, np.nan, np.nan], dtype=arr.real.dtype)
        sq_true = np.array([complex(np.nan, np.nan), complex(np.nan, np.nan), complex(np.nan, np.nan), complex(np.nan, np.inf), complex(-np.inf, np.nan), complex(np.inf, np.nan), complex(0.0, 0.0), complex(np.nan, np.nan), complex(np.nan, np.nan)], dtype=astype)
        with np.errstate(invalid='ignore'):
            assert_equal(np.abs(arr[::stride]), abs_true[::stride])
            assert_equal(np.square(arr[::stride]), sq_true[::stride])

class TestComplexAbsoluteAVX:

    @pytest.mark.parametrize('arraysize', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 18, 19])
    @pytest.mark.parametrize('stride', [-4, -3, -2, -1, 1, 2, 3, 4])
    @pytest.mark.parametrize('astype', [np.complex64, np.complex128])
    def test_array(self, arraysize, stride, astype):
        if False:
            print('Hello World!')
        arr = np.ones(arraysize, dtype=astype)
        abs_true = np.ones(arraysize, dtype=arr.real.dtype)
        assert_equal(np.abs(arr[::stride]), abs_true[::stride])

class TestComplexAbsoluteMixedDTypes:

    @pytest.mark.parametrize('stride', [-4, -3, -2, -1, 1, 2, 3, 4])
    @pytest.mark.parametrize('astype', [np.complex64, np.complex128])
    @pytest.mark.parametrize('func', ['abs', 'square', 'conjugate'])
    def test_array(self, stride, astype, func):
        if False:
            for i in range(10):
                print('nop')
        dtype = [('template_id', '<i8'), ('bank_chisq', '<f4'), ('bank_chisq_dof', '<i8'), ('chisq', '<f4'), ('chisq_dof', '<i8'), ('cont_chisq', '<f4'), ('psd_var_val', '<f4'), ('sg_chisq', '<f4'), ('mycomplex', astype), ('time_index', '<i8')]
        vec = np.array([(0, 0.0, 0, -31.666483, 200, 0.0, 0.0, 1.0, 3.0 + 4j, 613090), (1, 0.0, 0, 260.91525, 42, 0.0, 0.0, 1.0, 5.0 + 12j, 787315), (1, 0.0, 0, 52.15155, 42, 0.0, 0.0, 1.0, 8.0 + 15j, 806641), (1, 0.0, 0, 52.430195, 42, 0.0, 0.0, 1.0, 7.0 + 24j, 1363540), (2, 0.0, 0, 304.43646, 58, 0.0, 0.0, 1.0, 20.0 + 21j, 787323), (3, 0.0, 0, 299.42108, 52, 0.0, 0.0, 1.0, 12.0 + 35j, 787332), (4, 0.0, 0, 39.4836, 28, 0.0, 0.0, 9.182192, 9.0 + 40j, 787304), (4, 0.0, 0, 76.83787, 28, 0.0, 0.0, 1.0, 28.0 + 45j, 1321869), (5, 0.0, 0, 143.26366, 24, 0.0, 0.0, 10.996129, 11.0 + 60j, 787299)], dtype=dtype)
        myfunc = getattr(np, func)
        a = vec['mycomplex']
        g = myfunc(a[::stride])
        b = vec['mycomplex'].copy()
        h = myfunc(b[::stride])
        assert_array_max_ulp(h.real, g.real, 1)
        assert_array_max_ulp(h.imag, g.imag, 1)