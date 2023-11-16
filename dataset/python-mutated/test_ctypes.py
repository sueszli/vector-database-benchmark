from ctypes import *
import sys
import threading
import numpy as np
from numba.core.compiler import compile_isolated
from numba import jit
from numba.core import types, errors
from numba.core.typing import ctypes_utils
from numba.tests.support import MemoryLeakMixin, tag, TestCase
from numba.tests.ctypes_usecases import *
import unittest

class TestCTypesTypes(TestCase):

    def _conversion_tests(self, check):
        if False:
            return 10
        check(c_double, types.float64)
        check(c_int, types.intc)
        check(c_uint16, types.uint16)
        check(c_size_t, types.size_t)
        check(c_ssize_t, types.ssize_t)
        check(c_void_p, types.voidptr)
        check(POINTER(c_float), types.CPointer(types.float32))
        check(POINTER(POINTER(c_float)), types.CPointer(types.CPointer(types.float32)))
        check(None, types.void)

    def test_from_ctypes(self):
        if False:
            return 10
        '\n        Test converting a ctypes type to a Numba type.\n        '

        def check(cty, ty):
            if False:
                i = 10
                return i + 15
            got = ctypes_utils.from_ctypes(cty)
            self.assertEqual(got, ty)
        self._conversion_tests(check)
        with self.assertRaises(TypeError) as raises:
            ctypes_utils.from_ctypes(c_wchar_p)
        self.assertIn('Unsupported ctypes type', str(raises.exception))

    def test_to_ctypes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test converting a Numba type to a ctypes type.\n        '

        def check(cty, ty):
            if False:
                for i in range(10):
                    print('nop')
            got = ctypes_utils.to_ctypes(ty)
            self.assertEqual(got, cty)
        self._conversion_tests(check)
        with self.assertRaises(TypeError) as raises:
            ctypes_utils.to_ctypes(types.ellipsis)
        self.assertIn("Cannot convert Numba type '...' to ctypes type", str(raises.exception))

class TestCTypesUseCases(MemoryLeakMixin, TestCase):

    def test_c_sin(self):
        if False:
            i = 10
            return i + 15
        pyfunc = use_c_sin
        cres = compile_isolated(pyfunc, [types.double])
        cfunc = cres.entry_point
        x = 3.14
        self.assertEqual(pyfunc(x), cfunc(x))

    def test_two_funcs(self):
        if False:
            return 10
        pyfunc = use_two_funcs
        cres = compile_isolated(pyfunc, [types.double])
        cfunc = cres.entry_point
        x = 3.14
        self.assertEqual(pyfunc(x), cfunc(x))

    @unittest.skipUnless(is_windows, 'Windows-specific test')
    def test_stdcall(self):
        if False:
            i = 10
            return i + 15
        cres = compile_isolated(use_c_sleep, [types.uintc])
        cfunc = cres.entry_point
        cfunc(1)

    def test_ctype_wrapping(self):
        if False:
            while True:
                i = 10
        pyfunc = use_ctype_wrapping
        cres = compile_isolated(pyfunc, [types.double])
        cfunc = cres.entry_point
        x = 3.14
        self.assertEqual(pyfunc(x), cfunc(x))

    def test_ctype_voidptr(self):
        if False:
            return 10
        pyfunc = use_c_pointer
        cres = compile_isolated(pyfunc, [types.int32])
        cfunc = cres.entry_point
        x = 123
        self.assertEqual(cfunc(x), x + 1)

    def test_function_pointer(self):
        if False:
            print('Hello World!')
        pyfunc = use_func_pointer
        cfunc = jit(nopython=True)(pyfunc)
        for (fa, fb, x) in [(c_sin, c_cos, 1.0), (c_sin, c_cos, -1.0), (c_cos, c_sin, 1.0), (c_cos, c_sin, -1.0)]:
            expected = pyfunc(fa, fb, x)
            got = cfunc(fa, fb, x)
            self.assertEqual(got, expected)
        self.assertEqual(len(cfunc.overloads), 1, cfunc.overloads)

    def test_untyped_function(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError) as raises:
            compile_isolated(use_c_untyped, [types.double])
        self.assertIn("ctypes function '_numba_test_exp' doesn't define its argument types", str(raises.exception))

    def test_python_call_back(self):
        if False:
            for i in range(10):
                print('nop')
        mydct = {'what': 1232121}

        def call_me_maybe(arr):
            if False:
                for i in range(10):
                    print('nop')
            return mydct[arr[0].decode('ascii')]
        py_call_back = CFUNCTYPE(c_int, py_object)(call_me_maybe)

        def pyfunc(a):
            if False:
                while True:
                    i = 10
            what = py_call_back(a)
            return what
        cfunc = jit(nopython=True, nogil=True)(pyfunc)
        arr = np.array(['what'], dtype='S10')
        self.assertEqual(pyfunc(arr), cfunc(arr))

    def test_python_call_back_threaded(self):
        if False:
            print('Hello World!')

        def pyfunc(a, repeat):
            if False:
                i = 10
                return i + 15
            out = 0
            for _ in range(repeat):
                out += py_call_back(a)
            return out
        cfunc = jit(nopython=True, nogil=True)(pyfunc)
        arr = np.array(['what'], dtype='S10')
        repeat = 1000
        expected = pyfunc(arr, repeat)
        outputs = []
        cfunc(arr, repeat)

        def run(func, arr, repeat):
            if False:
                for i in range(10):
                    print('nop')
            outputs.append(func(arr, repeat))
        threads = [threading.Thread(target=run, args=(cfunc, arr, repeat)) for _ in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for got in outputs:
            self.assertEqual(expected, got)

    def test_passing_array_ctypes_data(self):
        if False:
            while True:
                i = 10
        '\n        Test the ".ctypes.data" attribute of an array can be passed\n        as a "void *" parameter.\n        '

        def pyfunc(arr):
            if False:
                print('Hello World!')
            return c_take_array_ptr(arr.ctypes.data)
        cfunc = jit(nopython=True, nogil=True)(pyfunc)
        arr = np.arange(5)
        expected = pyfunc(arr)
        got = cfunc(arr)
        self.assertEqual(expected, got)

    def check_array_ctypes(self, pyfunc):
        if False:
            while True:
                i = 10
        cfunc = jit(nopython=True)(pyfunc)
        arr = np.linspace(0, 10, 5)
        expected = arr ** 2.0
        got = cfunc(arr)
        self.assertPreciseEqual(expected, got)
        return cfunc

    def test_passing_array_ctypes_voidptr(self):
        if False:
            return 10
        '\n        Test the ".ctypes" attribute of an array can be passed\n        as a "void *" parameter.\n        '
        self.check_array_ctypes(use_c_vsquare)

    def test_passing_array_ctypes_voidptr_pass_ptr(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the ".ctypes" attribute of an array can be passed\n        as a pointer parameter of the right type.\n        '
        cfunc = self.check_array_ctypes(use_c_vcube)
        with self.assertRaises(errors.TypingError) as raises:
            cfunc(np.float32([0.0]))
        self.assertIn('No implementation of function ExternalFunctionPointer', str(raises.exception))

    def test_storing_voidptr_to_int_array(self):
        if False:
            while True:
                i = 10
        cproto = CFUNCTYPE(c_void_p)

        @cproto
        def get_voidstar():
            if False:
                for i in range(10):
                    print('nop')
            return 3735928559

        def pyfunc(a):
            if False:
                print('Hello World!')
            ptr = get_voidstar()
            a[0] = ptr
            return ptr
        cres = compile_isolated(pyfunc, [types.uintp[::1]])
        cfunc = cres.entry_point
        arr_got = np.zeros(1, dtype=np.uintp)
        arr_expect = arr_got.copy()
        ret_got = cfunc(arr_got)
        ret_expect = pyfunc(arr_expect)
        self.assertEqual(ret_expect, 3735928559)
        self.assertPreciseEqual(ret_got, ret_expect)
        self.assertPreciseEqual(arr_got, arr_expect)
if __name__ == '__main__':
    unittest.main()