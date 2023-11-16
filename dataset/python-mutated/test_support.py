import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest
DBL_EPSILON = 2 ** (-52)
FLT_EPSILON = 2 ** (-23)
INF = float('inf')
NAN = float('nan')

class TestAssertPreciseEqual(TestCase):
    """
    Tests for TestCase.assertPreciseEqual().
    """
    int_types = [int]
    np_float_types = [np.float32, np.float64]
    float_types = [float] + np_float_types
    np_complex_types = [np.complex64, np.complex128]
    complex_types = [complex] + np_complex_types
    bool_types = [bool, np.bool_]

    def eq(self, left, right, **kwargs):
        if False:
            for i in range(10):
                print('nop')

        def assert_succeed(left, right):
            if False:
                while True:
                    i = 10
            self.assertPreciseEqual(left, right, **kwargs)
            self.assertPreciseEqual(right, left, **kwargs)
        assert_succeed(left, right)
        assert_succeed((left, left), (right, right))
        assert_succeed([left, left], [right, right])

    def ne(self, left, right, **kwargs):
        if False:
            i = 10
            return i + 15

        def assert_fail(left, right):
            if False:
                i = 10
                return i + 15
            try:
                self.assertPreciseEqual(left, right, **kwargs)
            except AssertionError:
                pass
            else:
                self.fail('%s and %s unexpectedly considered equal' % (left, right))
        assert_fail(left, right)
        assert_fail(right, left)
        assert_fail((left, left), (right, right))
        assert_fail((right, right), (left, left))
        assert_fail([left, left], [right, right])
        assert_fail([right, right], [left, left])

    def test_types(self):
        if False:
            i = 10
            return i + 15
        for (i, f, c) in itertools.product(self.int_types, self.float_types, self.complex_types):
            self.ne(i(1), f(1))
            self.ne(f(1), c(1))
            self.ne(i(1), c(1))
        for (u, v) in itertools.product(self.int_types, self.int_types):
            self.eq(u(1), v(1))
        for (u, v) in itertools.product(self.int_types, self.bool_types):
            self.ne(u(1), v(1))
        for (u, v) in itertools.product(self.np_float_types, self.np_float_types):
            if u is v:
                self.eq(u(1), v(1))
            else:
                self.ne(u(1), v(1))
        for (u, v) in itertools.product(self.np_complex_types, self.np_complex_types):
            if u is v:
                self.eq(u(1), v(1))
            else:
                self.ne(u(1), v(1))

    def test_int_values(self):
        if False:
            while True:
                i = 10
        for tp in self.int_types:
            for prec in ['exact', 'single', 'double']:
                self.eq(tp(0), tp(0), prec=prec)
                self.ne(tp(0), tp(1), prec=prec)
                self.ne(tp(-1), tp(1), prec=prec)
                self.ne(tp(2 ** 80), tp(1 + 2 ** 80), prec=prec)

    def test_bool_values(self):
        if False:
            print('Hello World!')
        for (tpa, tpb) in itertools.product(self.bool_types, self.bool_types):
            self.eq(tpa(True), tpb(True))
            self.eq(tpa(False), tpb(False))
            self.ne(tpa(True), tpb(False))

    def test_abs_tol_parse(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            self.eq(np.float64(1e-17), np.float64(1e-17), abs_tol='invalid')
        with self.assertRaises(ValueError):
            self.eq(np.float64(1), np.float64(2), abs_tol=int(7))

    def test_float_values(self):
        if False:
            while True:
                i = 10
        for tp in self.float_types:
            for prec in ['exact', 'single', 'double']:
                self.eq(tp(1.5), tp(1.5), prec=prec)
                self.eq(tp(0.0), tp(0.0), prec=prec)
                self.eq(tp(-0.0), tp(-0.0), prec=prec)
                self.ne(tp(0.0), tp(-0.0), prec=prec)
                self.eq(tp(0.0), tp(-0.0), prec=prec, ignore_sign_on_zero=True)
                self.eq(tp(INF), tp(INF), prec=prec)
                self.ne(tp(INF), tp(1e+38), prec=prec)
                self.eq(tp(-INF), tp(-INF), prec=prec)
                self.ne(tp(INF), tp(-INF), prec=prec)
                self.eq(tp(NAN), tp(NAN), prec=prec)
                self.ne(tp(NAN), tp(0), prec=prec)
                self.ne(tp(NAN), tp(INF), prec=prec)
                self.ne(tp(NAN), tp(-INF), prec=prec)

    def test_float64_values(self):
        if False:
            for i in range(10):
                print('nop')
        for tp in [float, np.float64]:
            self.ne(tp(1.0 + DBL_EPSILON), tp(1.0))

    def test_float32_values(self):
        if False:
            print('Hello World!')
        tp = np.float32
        self.ne(tp(1.0 + FLT_EPSILON), tp(1.0))

    def test_float64_values_inexact(self):
        if False:
            for i in range(10):
                print('nop')
        for tp in [float, np.float64]:
            for scale in [1.0, -2 ** 3, 2 ** (-4), -2 ** (-20)]:
                a = scale * 1.0
                b = scale * (1.0 + DBL_EPSILON)
                c = scale * (1.0 + DBL_EPSILON * 2)
                d = scale * (1.0 + DBL_EPSILON * 4)
                self.ne(tp(a), tp(b))
                self.ne(tp(a), tp(b), prec='exact')
                self.eq(tp(a), tp(b), prec='double')
                self.eq(tp(a), tp(b), prec='double', ulps=1)
                self.ne(tp(a), tp(c), prec='double')
                self.eq(tp(a), tp(c), prec='double', ulps=2)
                self.ne(tp(a), tp(d), prec='double', ulps=2)
                self.eq(tp(a), tp(c), prec='double', ulps=3)
                self.eq(tp(a), tp(d), prec='double', ulps=3)
            self.eq(tp(1e-16), tp(3e-16), prec='double', abs_tol='eps')
            self.ne(tp(1e-16), tp(4e-16), prec='double', abs_tol='eps')
            self.eq(tp(1e-17), tp(1e-18), prec='double', abs_tol=1e-17)
            self.ne(tp(1e-17), tp(3e-17), prec='double', abs_tol=1e-17)

    def test_float32_values_inexact(self):
        if False:
            return 10
        tp = np.float32
        for scale in [1.0, -2 ** 3, 2 ** (-4), -2 ** (-20)]:
            a = scale * 1.0
            b = scale * (1.0 + FLT_EPSILON)
            c = scale * (1.0 + FLT_EPSILON * 2)
            d = scale * (1.0 + FLT_EPSILON * 4)
            self.ne(tp(a), tp(b))
            self.ne(tp(a), tp(b), prec='exact')
            self.ne(tp(a), tp(b), prec='double')
            self.eq(tp(a), tp(b), prec='single')
            self.ne(tp(a), tp(c), prec='single')
            self.eq(tp(a), tp(c), prec='single', ulps=2)
            self.ne(tp(a), tp(d), prec='single', ulps=2)
            self.eq(tp(a), tp(c), prec='single', ulps=3)
            self.eq(tp(a), tp(d), prec='single', ulps=3)
        self.eq(tp(1e-07), tp(2e-07), prec='single', abs_tol='eps')
        self.ne(tp(1e-07), tp(3e-07), prec='single', abs_tol='eps')
        self.eq(tp(1e-07), tp(1e-08), prec='single', abs_tol=1e-07)
        self.ne(tp(1e-07), tp(3e-07), prec='single', abs_tol=1e-07)

    def test_complex_values(self):
        if False:
            while True:
                i = 10
        (c_pp, c_pn, c_np, c_nn) = [complex(0.0, 0.0), complex(0.0, -0.0), complex(-0.0, 0.0), complex(-0.0, -0.0)]
        for tp in self.complex_types:
            for prec in ['exact', 'single', 'double']:
                self.eq(tp(1 + 2j), tp(1 + 2j), prec=prec)
                self.ne(tp(1 + 1j), tp(1 + 2j), prec=prec)
                self.ne(tp(2 + 2j), tp(1 + 2j), prec=prec)
                self.eq(tp(c_pp), tp(c_pp), prec=prec)
                self.eq(tp(c_np), tp(c_np), prec=prec)
                self.eq(tp(c_nn), tp(c_nn), prec=prec)
                self.ne(tp(c_pp), tp(c_pn), prec=prec)
                self.ne(tp(c_pn), tp(c_nn), prec=prec)
                self.eq(tp(complex(INF, INF)), tp(complex(INF, INF)), prec=prec)
                self.eq(tp(complex(INF, -INF)), tp(complex(INF, -INF)), prec=prec)
                self.eq(tp(complex(-INF, -INF)), tp(complex(-INF, -INF)), prec=prec)
                self.ne(tp(complex(INF, INF)), tp(complex(INF, -INF)), prec=prec)
                self.ne(tp(complex(INF, INF)), tp(complex(-INF, INF)), prec=prec)
                self.eq(tp(complex(INF, 0)), tp(complex(INF, 0)), prec=prec)
                self.eq(tp(complex(NAN, 0)), tp(complex(NAN, 0)), prec=prec)
                self.eq(tp(complex(0, NAN)), tp(complex(0, NAN)), prec=prec)
                self.eq(tp(complex(NAN, NAN)), tp(complex(NAN, NAN)), prec=prec)
                self.eq(tp(complex(INF, NAN)), tp(complex(INF, NAN)), prec=prec)
                self.eq(tp(complex(NAN, -INF)), tp(complex(NAN, -INF)), prec=prec)
            self.ne(tp(complex(INF, 0)), tp(complex(INF, 1)), prec='exact')

    def test_complex128_values_inexact(self):
        if False:
            while True:
                i = 10
        for tp in [complex, np.complex128]:
            for scale in [1.0, -2 ** 3, 2 ** (-4), -2 ** (-20)]:
                a = scale * 1.0
                b = scale * (1.0 + DBL_EPSILON)
                c = scale * (1.0 + DBL_EPSILON * 2)
                aa = tp(complex(a, a))
                ab = tp(complex(a, b))
                bb = tp(complex(b, b))
                self.ne(tp(aa), tp(ab))
                self.eq(tp(aa), tp(ab), prec='double')
                self.eq(tp(ab), tp(bb), prec='double')
                self.eq(tp(aa), tp(bb), prec='double')
                ac = tp(complex(a, c))
                cc = tp(complex(c, c))
                self.ne(tp(aa), tp(ac), prec='double')
                self.ne(tp(ac), tp(cc), prec='double')
                self.eq(tp(aa), tp(ac), prec='double', ulps=2)
                self.eq(tp(ac), tp(cc), prec='double', ulps=2)
                self.eq(tp(aa), tp(cc), prec='double', ulps=2)
                self.eq(tp(aa), tp(cc), prec='single')

    def test_complex64_values_inexact(self):
        if False:
            i = 10
            return i + 15
        tp = np.complex64
        for scale in [1.0, -2 ** 3, 2 ** (-4), -2 ** (-20)]:
            a = scale * 1.0
            b = scale * (1.0 + FLT_EPSILON)
            c = scale * (1.0 + FLT_EPSILON * 2)
            aa = tp(complex(a, a))
            ab = tp(complex(a, b))
            bb = tp(complex(b, b))
            self.ne(tp(aa), tp(ab))
            self.ne(tp(aa), tp(ab), prec='double')
            self.eq(tp(aa), tp(ab), prec='single')
            self.eq(tp(ab), tp(bb), prec='single')
            self.eq(tp(aa), tp(bb), prec='single')
            ac = tp(complex(a, c))
            cc = tp(complex(c, c))
            self.ne(tp(aa), tp(ac), prec='single')
            self.ne(tp(ac), tp(cc), prec='single')
            self.eq(tp(aa), tp(ac), prec='single', ulps=2)
            self.eq(tp(ac), tp(cc), prec='single', ulps=2)
            self.eq(tp(aa), tp(cc), prec='single', ulps=2)

    def test_enums(self):
        if False:
            while True:
                i = 10
        values = [Color.red, Color.green, Color.blue, Shake.mint, Shape.circle, Shape.square, Planet.EARTH, Planet.MERCURY]
        for val in values:
            self.eq(val, val)
            self.ne(val, val.value)
        for (a, b) in itertools.combinations(values, 2):
            self.ne(a, b)

    def test_arrays(self):
        if False:
            print('Hello World!')
        a = np.arange(1, 7, dtype=np.int16).reshape((2, 3))
        b = a.copy()
        self.eq(a, b)
        self.ne(a, b + 1)
        self.ne(a, b[:-1])
        self.ne(a, b.T)
        self.ne(a, b.astype(np.int32))
        self.ne(a, b.T.copy().T)
        self.ne(a, b.flatten())
        b.flags.writeable = False
        self.ne(a, b)
        a = np.arange(1, 3, dtype=np.float64)
        b = a * (1.0 + DBL_EPSILON)
        c = a * (1.0 + DBL_EPSILON * 2)
        self.ne(a, b)
        self.eq(a, b, prec='double')
        self.ne(a, c, prec='double')

    def test_npdatetime(self):
        if False:
            i = 10
            return i + 15
        a = np.datetime64('1900', 'Y')
        b = np.datetime64('1900', 'Y')
        c = np.datetime64('1900-01-01', 'D')
        d = np.datetime64('1901', 'Y')
        self.eq(a, b)
        self.ne(a, c)
        self.ne(a, d)

    def test_nptimedelta(self):
        if False:
            i = 10
            return i + 15
        a = np.timedelta64(1, 'h')
        b = np.timedelta64(1, 'h')
        c = np.timedelta64(60, 'm')
        d = np.timedelta64(2, 'h')
        self.eq(a, b)
        self.ne(a, c)
        self.ne(a, d)

class TestMisc(TestCase):

    def test_assertRefCount(self):
        if False:
            return 10
        x = 55.0
        y = 66.0
        l = []
        with self.assertRefCount(x, y):
            pass
        with self.assertRaises(AssertionError) as cm:
            with self.assertRefCount(x, y):
                l.append(y)
        self.assertIn('66', str(cm.exception))

    def test_forbid_codegen(self):
        if False:
            print('Hello World!')
        '\n        Test that forbid_codegen() prevents code generation using the @jit\n        decorator.\n        '

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return 1
        with forbid_codegen():
            with self.assertRaises(RuntimeError) as raises:
                cfunc = jit(nopython=True)(f)
                cfunc()
        self.assertIn('codegen forbidden by test case', str(raises.exception))
if __name__ == '__main__':
    unittest.main()