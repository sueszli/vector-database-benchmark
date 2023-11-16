import numpy as np
from numba.core.compiler import compile_isolated
from numba.core.errors import TypingError
from numba import njit
from numba.core import types
import struct
import unittest

def float_to_int(x):
    if False:
        for i in range(10):
            print('nop')
    return types.int32(x)

def int_to_float(x):
    if False:
        for i in range(10):
            print('nop')
    return types.float64(x) / 2

def float_to_unsigned(x):
    if False:
        i = 10
        return i + 15
    return types.uint32(x)

def float_to_complex(x):
    if False:
        print('Hello World!')
    return types.complex128(x)

def numpy_scalar_cast_error():
    if False:
        return 10
    np.int32(np.zeros((4,)))

class TestCasting(unittest.TestCase):

    def test_float_to_int(self):
        if False:
            while True:
                i = 10
        pyfunc = float_to_int
        cr = compile_isolated(pyfunc, [types.float32])
        cfunc = cr.entry_point
        self.assertEqual(cr.signature.return_type, types.int32)
        self.assertEqual(cfunc(12.3), pyfunc(12.3))
        self.assertEqual(cfunc(12.3), int(12.3))
        self.assertEqual(cfunc(-12.3), pyfunc(-12.3))
        self.assertEqual(cfunc(-12.3), int(-12.3))

    def test_int_to_float(self):
        if False:
            return 10
        pyfunc = int_to_float
        cr = compile_isolated(pyfunc, [types.int64])
        cfunc = cr.entry_point
        self.assertEqual(cr.signature.return_type, types.float64)
        self.assertEqual(cfunc(321), pyfunc(321))
        self.assertEqual(cfunc(321), 321.0 / 2)

    def test_float_to_unsigned(self):
        if False:
            return 10
        pyfunc = float_to_unsigned
        cr = compile_isolated(pyfunc, [types.float32])
        cfunc = cr.entry_point
        self.assertEqual(cr.signature.return_type, types.uint32)
        self.assertEqual(cfunc(3.21), pyfunc(3.21))
        self.assertEqual(cfunc(3.21), struct.unpack('I', struct.pack('i', 3))[0])

    def test_float_to_complex(self):
        if False:
            while True:
                i = 10
        pyfunc = float_to_complex
        cr = compile_isolated(pyfunc, [types.float64])
        cfunc = cr.entry_point
        self.assertEqual(cr.signature.return_type, types.complex128)
        self.assertEqual(cfunc(-3.21), pyfunc(-3.21))
        self.assertEqual(cfunc(-3.21), -3.21 + 0j)

    def test_array_to_array(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure this compiles.\n\n        Cast C to A array\n        '

        @njit('f8(f8[:])')
        def inner(x):
            if False:
                return 10
            return x[0]
        inner.disable_compile()

        @njit('f8(f8[::1])')
        def driver(x):
            if False:
                print('Hello World!')
            return inner(x)
        x = np.array([1234], dtype=np.float64)
        self.assertEqual(driver(x), x[0])
        self.assertEqual(len(inner.overloads), 1)

    def test_0darrayT_to_T(self):
        if False:
            print('Hello World!')

        @njit
        def inner(x):
            if False:
                while True:
                    i = 10
            return x.dtype.type(x)
        inputs = [(np.bool_, True), (np.float32, 12.3), (np.float64, 12.3), (np.int64, 12), (np.complex64, 2j + 3), (np.complex128, 2j + 3), (np.timedelta64, np.timedelta64(3, 'h')), (np.datetime64, np.datetime64('2016-01-01')), ('<U3', 'ABC')]
        for (T, inp) in inputs:
            x = np.array(inp, dtype=T)
            self.assertEqual(inner(x), x[()])

    def test_array_to_scalar(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure that a TypingError exception is raised if\n        user tries to convert numpy array to scalar\n        '
        with self.assertRaises(TypingError) as raises:
            compile_isolated(numpy_scalar_cast_error, ())
        self.assertIn('Casting array(float64, 1d, C) to int32 directly is unsupported.', str(raises.exception))

    def test_optional_to_optional(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test error due mishandling of Optional to Optional casting\n\n        Related issue: https://github.com/numba/numba/issues/1718\n        '
        opt_int = types.Optional(types.intp)
        opt_flt = types.Optional(types.float64)
        sig = opt_flt(opt_int)

        @njit(sig)
        def foo(a):
            if False:
                for i in range(10):
                    print('nop')
            return a
        self.assertEqual(foo(2), 2)
        self.assertIsNone(foo(None))
if __name__ == '__main__':
    unittest.main()