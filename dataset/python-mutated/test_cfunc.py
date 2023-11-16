"""
Tests for @cfunc and friends.
"""
import ctypes
import os
import subprocess
import sys
from collections import namedtuple
import numpy as np
from numba import cfunc, carray, farray, njit
from numba.core import types, typing, utils
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import TestCase, skip_unless_cffi, tag, captured_stderr
import unittest
from numba.np import numpy_support

def add_usecase(a, b):
    if False:
        print('Hello World!')
    return a + b

def div_usecase(a, b):
    if False:
        for i in range(10):
            print('nop')
    c = a / b
    return c

def square_usecase(a):
    if False:
        print('Hello World!')
    return a ** 2
add_sig = 'float64(float64, float64)'
div_sig = 'float64(int64, int64)'
square_sig = 'float64(float64)'

def objmode_usecase(a, b):
    if False:
        for i in range(10):
            print('nop')
    object()
    return a + b
CARRAY_USECASE_OUT_LEN = 8

def make_cfarray_usecase(func):
    if False:
        print('Hello World!')

    def cfarray_usecase(in_ptr, out_ptr, m, n):
        if False:
            for i in range(10):
                print('nop')
        in_ = func(in_ptr, (m, n))
        out = func(out_ptr, CARRAY_USECASE_OUT_LEN)
        out[0] = in_.ndim
        out[1:3] = in_.shape
        out[3:5] = in_.strides
        out[5] = in_.flags.c_contiguous
        out[6] = in_.flags.f_contiguous
        s = 0
        for (i, j) in np.ndindex(m, n):
            s += in_[i, j] * (i - j)
        out[7] = s
    return cfarray_usecase
carray_usecase = make_cfarray_usecase(carray)
farray_usecase = make_cfarray_usecase(farray)

def make_cfarray_dtype_usecase(func):
    if False:
        return 10

    def cfarray_usecase(in_ptr, out_ptr, m, n):
        if False:
            return 10
        in_ = func(in_ptr, (m, n), dtype=np.float32)
        out = func(out_ptr, CARRAY_USECASE_OUT_LEN, np.float32)
        out[0] = in_.ndim
        out[1:3] = in_.shape
        out[3:5] = in_.strides
        out[5] = in_.flags.c_contiguous
        out[6] = in_.flags.f_contiguous
        s = 0
        for (i, j) in np.ndindex(m, n):
            s += in_[i, j] * (i - j)
        out[7] = s
    return cfarray_usecase
carray_dtype_usecase = make_cfarray_dtype_usecase(carray)
farray_dtype_usecase = make_cfarray_dtype_usecase(farray)
carray_float32_usecase_sig = types.void(types.CPointer(types.float32), types.CPointer(types.float32), types.intp, types.intp)
carray_float64_usecase_sig = types.void(types.CPointer(types.float64), types.CPointer(types.float64), types.intp, types.intp)
carray_voidptr_usecase_sig = types.void(types.voidptr, types.voidptr, types.intp, types.intp)

class TestCFunc(TestCase):

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Basic usage and properties of a cfunc.\n        '
        f = cfunc(add_sig)(add_usecase)
        self.assertEqual(f.__name__, 'add_usecase')
        self.assertEqual(f.__qualname__, 'add_usecase')
        self.assertIs(f.__wrapped__, add_usecase)
        symbol = f.native_name
        self.assertIsInstance(symbol, str)
        self.assertIn('add_usecase', symbol)
        addr = f.address
        self.assertIsInstance(addr, int)
        ct = f.ctypes
        self.assertEqual(ctypes.cast(ct, ctypes.c_void_p).value, addr)
        self.assertPreciseEqual(ct(2.0, 3.5), 5.5)

    @skip_unless_cffi
    def test_cffi(self):
        if False:
            for i in range(10):
                print('nop')
        from numba.tests import cffi_usecases
        (ffi, lib) = cffi_usecases.load_inline_module()
        f = cfunc(square_sig)(square_usecase)
        res = lib._numba_test_funcptr(f.cffi)
        self.assertPreciseEqual(res, 2.25)

    def test_locals(self):
        if False:
            while True:
                i = 10
        f = cfunc(div_sig, locals={'c': types.int64})(div_usecase)
        self.assertPreciseEqual(f.ctypes(8, 3), 2.0)

    def test_errors(self):
        if False:
            return 10
        f = cfunc(div_sig)(div_usecase)
        with captured_stderr() as err:
            self.assertPreciseEqual(f.ctypes(5, 2), 2.5)
        self.assertEqual(err.getvalue(), '')
        with captured_stderr() as err:
            res = f.ctypes(5, 0)
            self.assertPreciseEqual(res, 0.0)
        err = err.getvalue()
        self.assertIn('ZeroDivisionError:', err)
        self.assertIn('Exception ignored', err)

    def test_llvm_ir(self):
        if False:
            for i in range(10):
                print('nop')
        f = cfunc(add_sig)(add_usecase)
        ir = f.inspect_llvm()
        self.assertIn(f.native_name, ir)
        self.assertIn('fadd double', ir)

    def test_object_mode(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Object mode is currently unsupported.\n        '
        with self.assertRaises(NotImplementedError):
            cfunc(add_sig, forceobj=True)(add_usecase)
        with self.assertTypingError() as raises:
            cfunc(add_sig)(objmode_usecase)
        self.assertIn("Untyped global name 'object'", str(raises.exception))

class TestCArray(TestCase):
    """
    Tests for carray() and farray().
    """

    def run_carray_usecase(self, pointer_factory, func):
        if False:
            i = 10
            return i + 15
        a = np.arange(10, 16).reshape((2, 3)).astype(np.float32)
        out = np.empty(CARRAY_USECASE_OUT_LEN, dtype=np.float32)
        func(pointer_factory(a), pointer_factory(out), *a.shape)
        return out

    def check_carray_usecase(self, pointer_factory, pyfunc, cfunc):
        if False:
            print('Hello World!')
        expected = self.run_carray_usecase(pointer_factory, pyfunc)
        got = self.run_carray_usecase(pointer_factory, cfunc)
        self.assertPreciseEqual(expected, got)

    def make_voidptr(self, arr):
        if False:
            while True:
                i = 10
        return arr.ctypes.data_as(ctypes.c_void_p)

    def make_float32_pointer(self, arr):
        if False:
            i = 10
            return i + 15
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def make_float64_pointer(self, arr):
        if False:
            i = 10
            return i + 15
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def check_carray_farray(self, func, order):
        if False:
            return 10

        def eq(got, expected):
            if False:
                print('Hello World!')
            self.assertPreciseEqual(got, expected)
            self.assertEqual(got.ctypes.data, expected.ctypes.data)
        base = np.arange(6).reshape((2, 3)).astype(np.float32).copy(order=order)
        a = func(self.make_float32_pointer(base), base.shape)
        eq(a, base)
        a = func(self.make_float32_pointer(base), base.size)
        eq(a, base.ravel('K'))
        a = func(self.make_float32_pointer(base), base.shape, base.dtype)
        eq(a, base)
        a = func(self.make_float32_pointer(base), base.shape, np.float32)
        eq(a, base)
        a = func(self.make_voidptr(base), base.shape, base.dtype)
        eq(a, base)
        a = func(self.make_voidptr(base), base.shape, np.int32)
        eq(a, base.view(np.int32))
        with self.assertRaises(TypeError):
            func(self.make_voidptr(base), base.shape)
        with self.assertRaises(TypeError):
            func(base.ctypes.data, base.shape)
        with self.assertRaises(TypeError) as raises:
            func(self.make_float32_pointer(base), base.shape, np.int32)
        self.assertIn("mismatching dtype 'int32' for pointer", str(raises.exception))

    def test_carray(self):
        if False:
            while True:
                i = 10
        '\n        Test pure Python carray().\n        '
        self.check_carray_farray(carray, 'C')

    def test_farray(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test pure Python farray().\n        '
        self.check_carray_farray(farray, 'F')

    def make_carray_sigs(self, formal_sig):
        if False:
            while True:
                i = 10
        '\n        Generate a bunch of concrete signatures by varying the width\n        and signedness of size arguments (see issue #1923).\n        '
        for actual_size in (types.intp, types.int32, types.intc, types.uintp, types.uint32, types.uintc):
            args = tuple((actual_size if a == types.intp else a for a in formal_sig.args))
            yield formal_sig.return_type(*args)

    def check_numba_carray_farray(self, usecase, dtype_usecase):
        if False:
            print('Hello World!')
        pyfunc = usecase
        for sig in self.make_carray_sigs(carray_float32_usecase_sig):
            f = cfunc(sig)(pyfunc)
            self.check_carray_usecase(self.make_float32_pointer, pyfunc, f.ctypes)
        pyfunc = dtype_usecase
        for sig in self.make_carray_sigs(carray_float32_usecase_sig):
            f = cfunc(sig)(pyfunc)
            self.check_carray_usecase(self.make_float32_pointer, pyfunc, f.ctypes)
        with self.assertTypingError() as raises:
            f = cfunc(carray_float64_usecase_sig)(pyfunc)
        self.assertIn("mismatching dtype 'float32' for pointer type 'float64*'", str(raises.exception))
        pyfunc = dtype_usecase
        for sig in self.make_carray_sigs(carray_voidptr_usecase_sig):
            f = cfunc(sig)(pyfunc)
            self.check_carray_usecase(self.make_float32_pointer, pyfunc, f.ctypes)

    def test_numba_carray(self):
        if False:
            print('Hello World!')
        '\n        Test Numba-compiled carray() against pure Python carray()\n        '
        self.check_numba_carray_farray(carray_usecase, carray_dtype_usecase)

    def test_numba_farray(self):
        if False:
            while True:
                i = 10
        '\n        Test Numba-compiled farray() against pure Python farray()\n        '
        self.check_numba_carray_farray(farray_usecase, farray_dtype_usecase)

@skip_unless_cffi
class TestCffiStruct(TestCase):
    c_source = '\ntypedef struct _big_struct {\n    int    i1;\n    float  f2;\n    double d3;\n    float  af4[9];\n} big_struct;\n\ntypedef struct _error {\n    int bits:4;\n} error;\n\ntypedef double (*myfunc)(big_struct*, size_t);\n'

    def get_ffi(self, src=c_source):
        if False:
            i = 10
            return i + 15
        from cffi import FFI
        ffi = FFI()
        ffi.cdef(src)
        return ffi

    def test_type_parsing(self):
        if False:
            print('Hello World!')
        ffi = self.get_ffi()
        big_struct = ffi.typeof('big_struct')
        nbtype = cffi_support.map_type(big_struct, use_record_dtype=True)
        self.assertIsInstance(nbtype, types.Record)
        self.assertEqual(len(nbtype), 4)
        self.assertEqual(nbtype.typeof('i1'), types.int32)
        self.assertEqual(nbtype.typeof('f2'), types.float32)
        self.assertEqual(nbtype.typeof('d3'), types.float64)
        self.assertEqual(nbtype.typeof('af4'), types.NestedArray(dtype=types.float32, shape=(9,)))
        myfunc = ffi.typeof('myfunc')
        sig = cffi_support.map_type(myfunc, use_record_dtype=True)
        self.assertIsInstance(sig, typing.Signature)
        self.assertEqual(sig.args[0], types.CPointer(nbtype))
        self.assertEqual(sig.args[1], types.uintp)
        self.assertEqual(sig.return_type, types.float64)

    def test_cfunc_callback(self):
        if False:
            for i in range(10):
                print('nop')
        ffi = self.get_ffi()
        big_struct = ffi.typeof('big_struct')
        nb_big_struct = cffi_support.map_type(big_struct, use_record_dtype=True)
        sig = cffi_support.map_type(ffi.typeof('myfunc'), use_record_dtype=True)

        @njit
        def calc(base):
            if False:
                for i in range(10):
                    print('nop')
            tmp = 0
            for i in range(base.size):
                elem = base[i]
                tmp += elem.i1 * elem.f2 / elem.d3
                tmp += base[i].af4.sum()
            return tmp

        @cfunc(sig)
        def foo(ptr, n):
            if False:
                i = 10
                return i + 15
            base = carray(ptr, n)
            return calc(base)
        mydata = ffi.new('big_struct[3]')
        ptr = ffi.cast('big_struct*', mydata)
        for i in range(3):
            ptr[i].i1 = i * 123
            ptr[i].f2 = i * 213
            ptr[i].d3 = (1 + i) * 213
            for j in range(9):
                ptr[i].af4[j] = i * 10 + j
        addr = int(ffi.cast('size_t', ptr))
        got = foo.ctypes(addr, 3)
        array = np.ndarray(buffer=ffi.buffer(mydata), dtype=numpy_support.as_dtype(nb_big_struct), shape=3)
        expect = calc(array)
        self.assertEqual(got, expect)

    def test_unsupport_bitsize(self):
        if False:
            while True:
                i = 10
        ffi = self.get_ffi()
        with self.assertRaises(ValueError) as raises:
            cffi_support.map_type(ffi.typeof('error'), use_record_dtype=True)
        self.assertEqual("field 'bits' has bitshift, this is not supported", str(raises.exception))
if __name__ == '__main__':
    unittest.main()