import itertools
import math
import sys
import warnings
import numpy as np
from numba.core.compiler import compile_isolated, Flags
from numba.core import utils, types
from numba.core.config import IS_WIN32, IS_32BITS
from numba.tests.support import TestCase, CompilationCache, tag
import unittest
from numba.np import numpy_support
enable_pyobj_flags = Flags()
enable_pyobj_flags.enable_pyobject = True
no_pyobj_flags = Flags()

def sin(x):
    if False:
        return 10
    return math.sin(x)

def cos(x):
    if False:
        print('Hello World!')
    return math.cos(x)

def tan(x):
    if False:
        for i in range(10):
            print('nop')
    return math.tan(x)

def sinh(x):
    if False:
        i = 10
        return i + 15
    return math.sinh(x)

def cosh(x):
    if False:
        return 10
    return math.cosh(x)

def tanh(x):
    if False:
        print('Hello World!')
    return math.tanh(x)

def asin(x):
    if False:
        i = 10
        return i + 15
    return math.asin(x)

def acos(x):
    if False:
        for i in range(10):
            print('nop')
    return math.acos(x)

def atan(x):
    if False:
        i = 10
        return i + 15
    return math.atan(x)

def atan2(y, x):
    if False:
        while True:
            i = 10
    return math.atan2(y, x)

def asinh(x):
    if False:
        i = 10
        return i + 15
    return math.asinh(x)

def acosh(x):
    if False:
        for i in range(10):
            print('nop')
    return math.acosh(x)

def atanh(x):
    if False:
        i = 10
        return i + 15
    return math.atanh(x)

def sqrt(x):
    if False:
        for i in range(10):
            print('nop')
    return math.sqrt(x)

def npy_sqrt(x):
    if False:
        i = 10
        return i + 15
    return np.sqrt(x)

def exp(x):
    if False:
        print('Hello World!')
    return math.exp(x)

def expm1(x):
    if False:
        while True:
            i = 10
    return math.expm1(x)

def log(x):
    if False:
        i = 10
        return i + 15
    return math.log(x)

def log1p(x):
    if False:
        return 10
    return math.log1p(x)

def log10(x):
    if False:
        print('Hello World!')
    return math.log10(x)

def floor(x):
    if False:
        return 10
    return math.floor(x)

def ceil(x):
    if False:
        return 10
    return math.ceil(x)

def trunc(x):
    if False:
        print('Hello World!')
    return math.trunc(x)

def isnan(x):
    if False:
        print('Hello World!')
    return math.isnan(x)

def isinf(x):
    if False:
        return 10
    return math.isinf(x)

def isfinite(x):
    if False:
        print('Hello World!')
    return math.isfinite(x)

def hypot(x, y):
    if False:
        while True:
            i = 10
    return math.hypot(x, y)

def degrees(x):
    if False:
        print('Hello World!')
    return math.degrees(x)

def radians(x):
    if False:
        print('Hello World!')
    return math.radians(x)

def erf(x):
    if False:
        for i in range(10):
            print('nop')
    return math.erf(x)

def erfc(x):
    if False:
        for i in range(10):
            print('nop')
    return math.erfc(x)

def gamma(x):
    if False:
        return 10
    return math.gamma(x)

def lgamma(x):
    if False:
        i = 10
        return i + 15
    return math.lgamma(x)

def pow(x, y):
    if False:
        i = 10
        return i + 15
    return math.pow(x, y)

def gcd(x, y):
    if False:
        i = 10
        return i + 15
    return math.gcd(x, y)

def copysign(x, y):
    if False:
        for i in range(10):
            print('nop')
    return math.copysign(x, y)

def frexp(x):
    if False:
        return 10
    return math.frexp(x)

def ldexp(x, e):
    if False:
        for i in range(10):
            print('nop')
    return math.ldexp(x, e)

def get_constants():
    if False:
        i = 10
        return i + 15
    return (math.pi, math.e)

class TestMathLib(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.ccache = CompilationCache()

    def test_constants(self):
        if False:
            print('Hello World!')
        self.run_nullary_func(get_constants, no_pyobj_flags)

    def run_unary(self, pyfunc, x_types, x_values, flags=enable_pyobj_flags, prec='exact', **kwargs):
        if False:
            i = 10
            return i + 15
        for (tx, vx) in zip(x_types, x_values):
            cr = self.ccache.compile(pyfunc, (tx,), flags=flags)
            cfunc = cr.entry_point
            got = cfunc(vx)
            expected = pyfunc(vx)
            actual_prec = 'single' if tx is types.float32 else prec
            msg = 'for input %r' % (vx,)
            self.assertPreciseEqual(got, expected, prec=actual_prec, msg=msg, **kwargs)

    def run_binary(self, pyfunc, x_types, x_values, y_values, flags=enable_pyobj_flags, prec='exact'):
        if False:
            for i in range(10):
                print('nop')
        for (ty, x, y) in zip(x_types, x_values, y_values):
            cr = self.ccache.compile(pyfunc, (ty, ty), flags=flags)
            cfunc = cr.entry_point
            got = cfunc(x, y)
            expected = pyfunc(x, y)
            actual_prec = 'single' if ty is types.float32 else prec
            msg = 'for inputs (%r, %r)' % (x, y)
            self.assertPreciseEqual(got, expected, prec=actual_prec, msg=msg)

    def check_predicate_func(self, pyfunc, flags=enable_pyobj_flags):
        if False:
            for i in range(10):
                print('nop')
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float32, types.float32, types.float64, types.float64, types.float64]
        x_values = [0, 0, 0, 0, 0, 0, float('inf'), 0.0, float('nan'), float('inf'), 0.0, float('nan')]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_sin(self, flags=enable_pyobj_flags):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = sin
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [-2, -1, -2, 2, 1, 2, 0.1, 0.2]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_sin_npm(self):
        if False:
            return 10
        self.test_sin(flags=no_pyobj_flags)

    @unittest.skipIf(sys.platform == 'win32', 'not exactly equal on win32 (issue #597)')
    def test_cos(self, flags=enable_pyobj_flags):
        if False:
            i = 10
            return i + 15
        pyfunc = cos
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [-2, -1, -2, 2, 1, 2, 0.1, 0.2]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_cos_npm(self):
        if False:
            print('Hello World!')
        self.test_cos(flags=no_pyobj_flags)

    def test_tan(self, flags=enable_pyobj_flags):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = tan
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [-2, -1, -2, 2, 1, 2, 0.1, 0.2]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_tan_npm(self):
        if False:
            return 10
        self.test_tan(flags=no_pyobj_flags)

    def test_sqrt(self, flags=enable_pyobj_flags):
        if False:
            i = 10
            return i + 15
        pyfunc = sqrt
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [2, 1, 2, 2, 1, 2, 0.1, 0.2]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_sqrt_npm(self):
        if False:
            print('Hello World!')
        self.test_sqrt(flags=no_pyobj_flags)

    def test_npy_sqrt(self, flags=enable_pyobj_flags):
        if False:
            while True:
                i = 10
        pyfunc = npy_sqrt
        x_values = [2, 1, 2, 2, 1, 2, 0.1, 0.2]
        x_types = [types.int16, types.uint16]
        self.run_unary(pyfunc, x_types, x_values, flags, prec='single')
        x_types = [types.int32, types.int64, types.uint32, types.uint64, types.float32, types.float64]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_npy_sqrt_npm(self):
        if False:
            while True:
                i = 10
        self.test_npy_sqrt(flags=no_pyobj_flags)

    def test_exp(self, flags=enable_pyobj_flags):
        if False:
            i = 10
            return i + 15
        pyfunc = exp
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [-2, -1, -2, 2, 1, 2, 0.1, 0.2]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_exp_npm(self):
        if False:
            return 10
        self.test_exp(flags=no_pyobj_flags)

    def test_expm1(self, flags=enable_pyobj_flags):
        if False:
            print('Hello World!')
        pyfunc = expm1
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [-2, -1, -2, 2, 1, 2, 0.1, 0.2]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_expm1_npm(self):
        if False:
            i = 10
            return i + 15
        self.test_expm1(flags=no_pyobj_flags)

    def test_log(self, flags=enable_pyobj_flags):
        if False:
            i = 10
            return i + 15
        pyfunc = log
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [1, 10, 100, 1000, 100000, 1000000, 0.1, 1.1]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_log_npm(self):
        if False:
            return 10
        self.test_log(flags=no_pyobj_flags)

    def test_log1p(self, flags=enable_pyobj_flags):
        if False:
            i = 10
            return i + 15
        pyfunc = log1p
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [1, 10, 100, 1000, 100000, 1000000, 0.1, 1.1]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_log1p_npm(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_log1p(flags=no_pyobj_flags)

    def test_log10(self, flags=enable_pyobj_flags):
        if False:
            print('Hello World!')
        pyfunc = log10
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [1, 10, 100, 1000, 100000, 1000000, 0.1, 1.1]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_log10_npm(self):
        if False:
            i = 10
            return i + 15
        self.test_log10(flags=no_pyobj_flags)

    def test_asin(self, flags=enable_pyobj_flags):
        if False:
            print('Hello World!')
        pyfunc = asin
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [1, 1, 1, 1, 1, 1, 1.0, 1.0]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_asin_npm(self):
        if False:
            i = 10
            return i + 15
        self.test_asin(flags=no_pyobj_flags)

    def test_acos(self, flags=enable_pyobj_flags):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = acos
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [1, 1, 1, 1, 1, 1, 1.0, 1.0]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_acos_npm(self):
        if False:
            while True:
                i = 10
        self.test_acos(flags=no_pyobj_flags)

    def test_atan(self, flags=enable_pyobj_flags):
        if False:
            print('Hello World!')
        pyfunc = atan
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [-2, -1, -2, 2, 1, 2, 0.1, 0.2]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_atan_npm(self):
        if False:
            while True:
                i = 10
        self.test_atan(flags=no_pyobj_flags)

    def test_atan2(self, flags=enable_pyobj_flags):
        if False:
            return 10
        pyfunc = atan2
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [-2, -1, -2, 2, 1, 2, 0.1, 0.2]
        y_values = [x * 2 for x in x_values]
        self.run_binary(pyfunc, x_types, x_values, y_values, flags)

    def test_atan2_npm(self):
        if False:
            return 10
        self.test_atan2(flags=no_pyobj_flags)

    def test_asinh(self, flags=enable_pyobj_flags):
        if False:
            return 10
        pyfunc = asinh
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [1, 1, 1, 1, 1, 1, 1.0, 1.0]
        self.run_unary(pyfunc, x_types, x_values, flags, prec='double')

    def test_asinh_npm(self):
        if False:
            return 10
        self.test_asinh(flags=no_pyobj_flags)

    def test_acosh(self, flags=enable_pyobj_flags):
        if False:
            i = 10
            return i + 15
        pyfunc = acosh
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [1, 1, 1, 1, 1, 1, 1.0, 1.0]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_acosh_npm(self):
        if False:
            return 10
        self.test_acosh(flags=no_pyobj_flags)

    def test_atanh(self, flags=enable_pyobj_flags):
        if False:
            print('Hello World!')
        pyfunc = atanh
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [0, 0, 0, 0, 0, 0, 0.1, 0.1]
        self.run_unary(pyfunc, x_types, x_values, flags, prec='double')

    def test_atanh_npm(self):
        if False:
            while True:
                i = 10
        self.test_atanh(flags=no_pyobj_flags)

    def test_sinh(self, flags=enable_pyobj_flags):
        if False:
            print('Hello World!')
        pyfunc = sinh
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [1, 1, 1, 1, 1, 1, 1.0, 1.0]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_sinh_npm(self):
        if False:
            i = 10
            return i + 15
        self.test_sinh(flags=no_pyobj_flags)

    def test_cosh(self, flags=enable_pyobj_flags):
        if False:
            print('Hello World!')
        pyfunc = cosh
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [1, 1, 1, 1, 1, 1, 1.0, 1.0]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_cosh_npm(self):
        if False:
            return 10
        self.test_cosh(flags=no_pyobj_flags)

    def test_tanh(self, flags=enable_pyobj_flags):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = tanh
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [0, 0, 0, 0, 0, 0, 0.1, 0.1]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_tanh_npm(self):
        if False:
            i = 10
            return i + 15
        self.test_tanh(flags=no_pyobj_flags)

    def test_floor(self, flags=enable_pyobj_flags):
        if False:
            while True:
                i = 10
        pyfunc = floor
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [0, 0, 0, 0, 0, 0, 0.1, 1.9]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_floor_npm(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_floor(flags=no_pyobj_flags)

    def test_ceil(self, flags=enable_pyobj_flags):
        if False:
            return 10
        pyfunc = ceil
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [0, 0, 0, 0, 0, 0, 0.1, 1.9]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_ceil_npm(self):
        if False:
            return 10
        self.test_ceil(flags=no_pyobj_flags)

    def test_trunc(self, flags=enable_pyobj_flags):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = trunc
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [0, 0, 0, 0, 0, 0, 0.1, 1.9]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_trunc_npm(self):
        if False:
            while True:
                i = 10
        self.test_trunc(flags=no_pyobj_flags)

    def test_isnan(self):
        if False:
            while True:
                i = 10
        self.check_predicate_func(isnan, flags=enable_pyobj_flags)

    def test_isnan_npm(self):
        if False:
            i = 10
            return i + 15
        self.check_predicate_func(isnan, flags=no_pyobj_flags)

    def test_isinf(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_predicate_func(isinf, flags=enable_pyobj_flags)

    def test_isinf_npm(self):
        if False:
            i = 10
            return i + 15
        self.check_predicate_func(isinf, flags=no_pyobj_flags)

    def test_isfinite(self):
        if False:
            i = 10
            return i + 15
        self.check_predicate_func(isfinite, flags=enable_pyobj_flags)

    def test_isfinite_npm(self):
        if False:
            while True:
                i = 10
        self.check_predicate_func(isfinite, flags=no_pyobj_flags)

    def test_hypot(self, flags=enable_pyobj_flags):
        if False:
            return 10
        pyfunc = hypot
        x_types = [types.int64, types.uint64, types.float32, types.float64]
        x_values = [1, 2, 3, 4, 5, 6, 0.21, 0.34]
        y_values = [x + 2 for x in x_values]
        prec = 'single'
        self.run_binary(pyfunc, x_types, x_values, y_values, flags, prec)

        def naive_hypot(x, y):
            if False:
                return 10
            return math.sqrt(x * x + y * y)
        for fltty in (types.float32, types.float64):
            cr = self.ccache.compile(pyfunc, (fltty, fltty), flags=flags)
            cfunc = cr.entry_point
            dt = numpy_support.as_dtype(fltty).type
            val = dt(np.finfo(dt).max / 30.0)
            nb_ans = cfunc(val, val)
            self.assertPreciseEqual(nb_ans, pyfunc(val, val), prec='single')
            self.assertTrue(np.isfinite(nb_ans))
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                self.assertRaisesRegexp(RuntimeWarning, 'overflow encountered in .*scalar', naive_hypot, val, val)

    def test_hypot_npm(self):
        if False:
            print('Hello World!')
        self.test_hypot(flags=no_pyobj_flags)

    def test_degrees(self, flags=enable_pyobj_flags):
        if False:
            return 10
        pyfunc = degrees
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [1, 1, 1, 1, 1, 1, 1.0, 1.0]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_degrees_npm(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_degrees(flags=no_pyobj_flags)

    def test_radians(self, flags=enable_pyobj_flags):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = radians
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [1, 1, 1, 1, 1, 1, 1.0, 1.0]
        self.run_unary(pyfunc, x_types, x_values, flags)

    def test_radians_npm(self):
        if False:
            i = 10
            return i + 15
        self.test_radians(flags=no_pyobj_flags)

    def test_erf(self, flags=enable_pyobj_flags):
        if False:
            while True:
                i = 10
        pyfunc = erf
        x_values = [1.0, 1.0, -1.0, -0.0, 0.0, 0.5, 5, float('inf')]
        x_types = [types.float32, types.float64] * (len(x_values) // 2)
        self.run_unary(pyfunc, x_types, x_values, flags, prec='double', ulps=2)

    def test_erf_npm(self):
        if False:
            while True:
                i = 10
        self.test_erf(flags=no_pyobj_flags)

    def test_erfc(self, flags=enable_pyobj_flags):
        if False:
            while True:
                i = 10
        pyfunc = erfc
        x_values = [1.0, 1.0, -1.0, -0.0, 0.0, 0.5, 5, float('inf')]
        x_types = [types.float32, types.float64] * (len(x_values) // 2)
        self.run_unary(pyfunc, x_types, x_values, flags, prec='double', ulps=4)

    def test_erfc_npm(self):
        if False:
            return 10
        self.test_erfc(flags=no_pyobj_flags)

    def test_gamma(self, flags=enable_pyobj_flags):
        if False:
            i = 10
            return i + 15
        pyfunc = gamma
        x_values = [1.0, -0.9, -0.5, 0.5]
        x_types = [types.float32, types.float64] * (len(x_values) // 2)
        self.run_unary(pyfunc, x_types, x_values, flags, prec='double', ulps=3)
        x_values = [-0.1, 0.1, 2.5, 10.1, 50.0, float('inf')]
        x_types = [types.float64] * len(x_values)
        self.run_unary(pyfunc, x_types, x_values, flags, prec='double', ulps=8)

    def test_gamma_npm(self):
        if False:
            print('Hello World!')
        self.test_gamma(flags=no_pyobj_flags)

    def test_lgamma(self, flags=enable_pyobj_flags):
        if False:
            i = 10
            return i + 15
        pyfunc = lgamma
        x_values = [1.0, -0.9, -0.1, 0.1, 200.0, 10000000000.0, 1e+30, float('inf')]
        x_types = [types.float32, types.float64] * (len(x_values) // 2)
        self.run_unary(pyfunc, x_types, x_values, flags, prec='double')

    def test_lgamma_npm(self):
        if False:
            return 10
        self.test_lgamma(flags=no_pyobj_flags)

    def test_pow(self, flags=enable_pyobj_flags):
        if False:
            while True:
                i = 10
        pyfunc = pow
        x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
        x_values = [-2, -1, -2, 2, 1, 2, 0.1, 0.2]
        y_values = [x * 2 for x in x_values]
        self.run_binary(pyfunc, x_types, x_values, y_values, flags)

    def test_gcd(self, flags=enable_pyobj_flags):
        if False:
            while True:
                i = 10
        from itertools import product, repeat, chain
        pyfunc = gcd
        signed_args = product(sorted(types.signed_domain), *repeat((-2, -1, 0, 1, 2, 7, 10), 2))
        unsigned_args = product(sorted(types.unsigned_domain), *repeat((0, 1, 2, 7, 9, 16), 2))
        (x_types, x_values, y_values) = zip(*chain(signed_args, unsigned_args))
        self.run_binary(pyfunc, x_types, x_values, y_values, flags)

    def test_gcd_npm(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_gcd(flags=no_pyobj_flags)

    def test_pow_npm(self):
        if False:
            i = 10
            return i + 15
        self.test_pow(flags=no_pyobj_flags)

    def test_copysign(self, flags=enable_pyobj_flags):
        if False:
            while True:
                i = 10
        pyfunc = copysign
        value_types = [types.float32, types.float64]
        values = [-2, -1, -0.0, 0.0, 1, 2, float('-inf'), float('inf'), float('nan')]
        (x_types, x_values, y_values) = list(zip(*itertools.product(value_types, values, values)))
        self.run_binary(pyfunc, x_types, x_values, y_values, flags)

    def test_copysign_npm(self):
        if False:
            while True:
                i = 10
        self.test_copysign(flags=no_pyobj_flags)

    def test_frexp(self, flags=enable_pyobj_flags):
        if False:
            while True:
                i = 10
        pyfunc = frexp
        x_types = [types.float32, types.float64]
        x_values = [-2.5, -0.0, 0.0, 3.5, float('-inf'), float('inf'), float('nan')]
        self.run_unary(pyfunc, x_types, x_values, flags, prec='exact')

    def test_frexp_npm(self):
        if False:
            return 10
        self.test_frexp(flags=no_pyobj_flags)

    def test_ldexp(self, flags=enable_pyobj_flags):
        if False:
            while True:
                i = 10
        pyfunc = ldexp
        for fltty in (types.float32, types.float64):
            cr = self.ccache.compile(pyfunc, (fltty, types.int32), flags=flags)
            cfunc = cr.entry_point
            for args in [(2.5, -2), (2.5, 1), (0.0, 0), (0.0, 1), (-0.0, 0), (-0.0, 1), (float('inf'), 0), (float('-inf'), 0), (float('nan'), 0)]:
                msg = 'for input %r' % (args,)
                self.assertPreciseEqual(cfunc(*args), pyfunc(*args))

    def test_ldexp_npm(self):
        if False:
            i = 10
            return i + 15
        self.test_ldexp(flags=no_pyobj_flags)
if __name__ == '__main__':
    unittest.main()