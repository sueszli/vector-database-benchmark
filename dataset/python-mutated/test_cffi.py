import array
import numpy as np
from numba import jit
import numba.core.typing.cffi_utils as cffi_support
from numba.core import types, errors
from numba.core.compiler import compile_isolated, Flags
from numba.tests.support import TestCase, skip_unless_cffi, tag
import numba.tests.cffi_usecases as mod
import unittest
enable_pyobj_flags = Flags()
enable_pyobj_flags.enable_pyobject = True
no_pyobj_flags = Flags()

@skip_unless_cffi
class TestCFFI(TestCase):
    _numba_parallel_test_ = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        mod.init()
        mod.init_ool()

    def test_type_map(self):
        if False:
            while True:
                i = 10
        signature = cffi_support.map_type(mod.ffi.typeof(mod.cffi_sin))
        self.assertEqual(len(signature.args), 1)
        self.assertEqual(signature.args[0], types.double)

    def _test_function(self, pyfunc, flags=enable_pyobj_flags):
        if False:
            return 10
        cres = compile_isolated(pyfunc, [types.double], flags=flags)
        cfunc = cres.entry_point
        for x in [-1.2, -1, 0, 0.1, 3.14]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_sin_function(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_function(mod.use_cffi_sin)

    def test_bool_function_ool(self):
        if False:
            print('Hello World!')
        pyfunc = mod.use_cffi_boolean_true
        cres = compile_isolated(pyfunc, (), flags=no_pyobj_flags)
        cfunc = cres.entry_point
        self.assertEqual(pyfunc(), True)
        self.assertEqual(cfunc(), True)

    def test_sin_function_npm(self):
        if False:
            while True:
                i = 10
        self._test_function(mod.use_cffi_sin, flags=no_pyobj_flags)

    def test_sin_function_ool(self, flags=enable_pyobj_flags):
        if False:
            i = 10
            return i + 15
        self._test_function(mod.use_cffi_sin_ool)

    def test_sin_function_npm_ool(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_function(mod.use_cffi_sin_ool, flags=no_pyobj_flags)

    def test_two_funcs(self):
        if False:
            while True:
                i = 10
        self._test_function(mod.use_two_funcs)

    def test_two_funcs_ool(self):
        if False:
            i = 10
            return i + 15
        self._test_function(mod.use_two_funcs_ool)

    def test_function_pointer(self):
        if False:
            i = 10
            return i + 15
        pyfunc = mod.use_func_pointer
        cfunc = jit(nopython=True)(pyfunc)
        for (fa, fb, x) in [(mod.cffi_sin, mod.cffi_cos, 1.0), (mod.cffi_sin, mod.cffi_cos, -1.0), (mod.cffi_cos, mod.cffi_sin, 1.0), (mod.cffi_cos, mod.cffi_sin, -1.0), (mod.cffi_sin_ool, mod.cffi_cos_ool, 1.0), (mod.cffi_sin_ool, mod.cffi_cos_ool, -1.0), (mod.cffi_cos_ool, mod.cffi_sin_ool, 1.0), (mod.cffi_cos_ool, mod.cffi_sin_ool, -1.0), (mod.cffi_sin, mod.cffi_cos_ool, 1.0), (mod.cffi_sin, mod.cffi_cos_ool, -1.0), (mod.cffi_cos, mod.cffi_sin_ool, 1.0), (mod.cffi_cos, mod.cffi_sin_ool, -1.0)]:
            expected = pyfunc(fa, fb, x)
            got = cfunc(fa, fb, x)
            self.assertEqual(got, expected)
        self.assertEqual(len(cfunc.overloads), 1, cfunc.overloads)

    def test_user_defined_symbols(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = mod.use_user_defined_symbols
        cfunc = jit(nopython=True)(pyfunc)
        self.assertEqual(pyfunc(), cfunc())

    def check_vector_sin(self, cfunc, x, y):
        if False:
            print('Hello World!')
        cfunc(x, y)
        np.testing.assert_allclose(y, np.sin(x))

    def _test_from_buffer_numpy_array(self, pyfunc, dtype):
        if False:
            return 10
        x = np.arange(10).astype(dtype)
        y = np.zeros_like(x)
        cfunc = jit(nopython=True)(pyfunc)
        self.check_vector_sin(cfunc, x, y)

    def test_from_buffer_float32(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_from_buffer_numpy_array(mod.vector_sin_float32, np.float32)

    def test_from_buffer_float64(self):
        if False:
            print('Hello World!')
        self._test_from_buffer_numpy_array(mod.vector_sin_float64, np.float64)

    def test_from_buffer_struct(self):
        if False:
            i = 10
            return i + 15
        n = 10
        x = np.arange(n) + np.arange(n * 2, n * 3) * 1j
        y = np.zeros(n)
        real_cfunc = jit(nopython=True)(mod.vector_extract_real)
        real_cfunc(x, y)
        np.testing.assert_equal(x.real, y)
        imag_cfunc = jit(nopython=True)(mod.vector_extract_imag)
        imag_cfunc(x, y)
        np.testing.assert_equal(x.imag, y)

    def test_from_buffer_pyarray(self):
        if False:
            print('Hello World!')
        pyfunc = mod.vector_sin_float32
        cfunc = jit(nopython=True)(pyfunc)
        x = array.array('f', range(10))
        y = array.array('f', [0] * len(x))
        self.check_vector_sin(cfunc, x, y)

    def test_from_buffer_error(self):
        if False:
            i = 10
            return i + 15
        pyfunc = mod.vector_sin_float32
        cfunc = jit(nopython=True)(pyfunc)
        x = np.arange(10).astype(np.float32)[::2]
        y = np.zeros_like(x)
        with self.assertRaises(errors.TypingError) as raises:
            cfunc(x, y)
        self.assertIn('from_buffer() unsupported on non-contiguous buffers', str(raises.exception))

    def test_from_buffer_numpy_multi_array(self):
        if False:
            print('Hello World!')
        c1 = np.array([1, 2], order='C', dtype=np.float32)
        c1_zeros = np.zeros_like(c1)
        c2 = np.array([[1, 2], [3, 4]], order='C', dtype=np.float32)
        c2_zeros = np.zeros_like(c2)
        f1 = np.array([1, 2], order='F', dtype=np.float32)
        f1_zeros = np.zeros_like(f1)
        f2 = np.array([[1, 2], [3, 4]], order='F', dtype=np.float32)
        f2_zeros = np.zeros_like(f2)
        f2_copy = f2.copy('K')
        pyfunc = mod.vector_sin_float32
        cfunc = jit(nopython=True)(pyfunc)
        self.check_vector_sin(cfunc, c1, c1_zeros)
        cfunc(c2, c2_zeros)
        sin_c2 = np.sin(c2)
        sin_c2[1] = [0, 0]
        np.testing.assert_allclose(c2_zeros, sin_c2)
        self.check_vector_sin(cfunc, f1, f1_zeros)
        with self.assertRaises(errors.TypingError) as raises:
            cfunc(f2, f2_zeros)
        np.testing.assert_allclose(f2, f2_copy)
        self.assertIn('from_buffer() only supports multidimensional arrays with C layout', str(raises.exception))

    def test_indirect_multiple_use(self):
        if False:
            i = 10
            return i + 15
        '\n        Issue #2263\n\n        Linkage error due to multiple definition of global tracking symbol.\n        '
        my_sin = mod.cffi_sin

        @jit(nopython=True)
        def inner(x):
            if False:
                while True:
                    i = 10
            return my_sin(x)

        @jit(nopython=True)
        def foo(x):
            if False:
                print('Hello World!')
            return inner(x) + my_sin(x + 1)
        x = 1.123
        self.assertEqual(foo(x), my_sin(x) + my_sin(x + 1))
if __name__ == '__main__':
    unittest.main()