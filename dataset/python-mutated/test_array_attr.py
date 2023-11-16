import numpy as np
import unittest
from numba.core.compiler import compile_isolated
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import TestCase, CompilationCache, MemoryLeakMixin, tag, skip_parfors_unsupported
from numba.core.errors import TypingError
from numba.experimental import jitclass

def array_dtype(a):
    if False:
        print('Hello World!')
    return a.dtype

def use_dtype(a, b):
    if False:
        while True:
            i = 10
    return a.view(b.dtype)

def array_itemsize(a):
    if False:
        for i in range(10):
            print('nop')
    return a.itemsize

def array_nbytes(a):
    if False:
        for i in range(10):
            print('nop')
    return a.nbytes

def array_shape(a, i):
    if False:
        while True:
            i = 10
    return a.shape[i]

def array_strides(a, i):
    if False:
        while True:
            i = 10
    return a.strides[i]

def array_ndim(a):
    if False:
        while True:
            i = 10
    return a.ndim

def array_size(a):
    if False:
        for i in range(10):
            print('nop')
    return a.size

def array_flags_contiguous(a):
    if False:
        i = 10
        return i + 15
    return a.flags.contiguous

def array_flags_c_contiguous(a):
    if False:
        return 10
    return a.flags.c_contiguous

def array_flags_f_contiguous(a):
    if False:
        print('Hello World!')
    return a.flags.f_contiguous

def nested_array_itemsize(a):
    if False:
        return 10
    return a.f.itemsize

def nested_array_nbytes(a):
    if False:
        i = 10
        return i + 15
    return a.f.nbytes

def nested_array_shape(a):
    if False:
        print('Hello World!')
    return a.f.shape

def nested_array_strides(a):
    if False:
        return 10
    return a.f.strides

def nested_array_ndim(a):
    if False:
        for i in range(10):
            print('nop')
    return a.f.ndim

def nested_array_size(a):
    if False:
        return 10
    return a.f.size

def size_after_slicing_usecase(buf, i):
    if False:
        return 10
    sliced = buf[i]
    return sliced.size

def array_ctypes_data(arr):
    if False:
        while True:
            i = 10
    return arr.ctypes.data

def array_real(arr):
    if False:
        print('Hello World!')
    return arr.real

def array_imag(arr):
    if False:
        for i in range(10):
            print('nop')
    return arr.imag

class TestArrayAttr(MemoryLeakMixin, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestArrayAttr, self).setUp()
        self.ccache = CompilationCache()
        self.a = np.arange(20, dtype=np.int32).reshape(4, 5)

    def check_unary(self, pyfunc, arr):
        if False:
            i = 10
            return i + 15
        aryty = typeof(arr)
        cfunc = self.get_cfunc(pyfunc, (aryty,))
        expected = pyfunc(arr)
        self.assertPreciseEqual(cfunc(arr), expected)
        cfunc = self.get_cfunc(pyfunc, (aryty.copy(layout='A'),))
        self.assertPreciseEqual(cfunc(arr), expected)

    def check_unary_with_arrays(self, pyfunc):
        if False:
            return 10
        self.check_unary(pyfunc, self.a)
        self.check_unary(pyfunc, self.a.T)
        self.check_unary(pyfunc, self.a[::2])
        arr = np.array([42]).reshape(())
        self.check_unary(pyfunc, arr)
        arr = np.zeros(0)
        self.check_unary(pyfunc, arr)
        self.check_unary(pyfunc, arr.reshape((1, 0, 2)))

    def get_cfunc(self, pyfunc, argspec):
        if False:
            while True:
                i = 10
        cres = self.ccache.compile(pyfunc, argspec)
        return cres.entry_point

    def test_shape(self):
        if False:
            return 10
        pyfunc = array_shape
        cfunc = self.get_cfunc(pyfunc, (types.int32[:, :], types.int32))
        for i in range(self.a.ndim):
            self.assertEqual(pyfunc(self.a, i), cfunc(self.a, i))

    def test_strides(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = array_strides
        cfunc = self.get_cfunc(pyfunc, (types.int32[:, :], types.int32))
        for i in range(self.a.ndim):
            self.assertEqual(pyfunc(self.a, i), cfunc(self.a, i))

    def test_ndim(self):
        if False:
            print('Hello World!')
        self.check_unary_with_arrays(array_ndim)

    def test_size(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_unary_with_arrays(array_size)

    def test_itemsize(self):
        if False:
            return 10
        self.check_unary_with_arrays(array_itemsize)

    def test_nbytes(self):
        if False:
            i = 10
            return i + 15
        self.check_unary_with_arrays(array_nbytes)

    def test_dtype(self):
        if False:
            return 10
        pyfunc = array_dtype
        self.check_unary(pyfunc, self.a)
        dtype = np.dtype([('x', np.int8), ('y', np.int8)])
        arr = np.zeros(4, dtype=dtype)
        self.check_unary(pyfunc, arr)

    def test_use_dtype(self):
        if False:
            return 10
        b = np.empty(1, dtype=np.int16)
        pyfunc = use_dtype
        cfunc = self.get_cfunc(pyfunc, (typeof(self.a), typeof(b)))
        expected = pyfunc(self.a, b)
        self.assertPreciseEqual(cfunc(self.a, b), expected)

    def test_flags_contiguous(self):
        if False:
            print('Hello World!')
        self.check_unary_with_arrays(array_flags_contiguous)

    def test_flags_c_contiguous(self):
        if False:
            return 10
        self.check_unary_with_arrays(array_flags_c_contiguous)

    def test_flags_f_contiguous(self):
        if False:
            while True:
                i = 10
        self.check_unary_with_arrays(array_flags_f_contiguous)

class TestNestedArrayAttr(MemoryLeakMixin, unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestNestedArrayAttr, self).setUp()
        dtype = np.dtype([('a', np.int32), ('f', np.int32, (2, 5))])
        self.a = np.recarray(1, dtype)[0]
        self.nbrecord = from_dtype(self.a.dtype)

    def get_cfunc(self, pyfunc):
        if False:
            i = 10
            return i + 15
        cres = compile_isolated(pyfunc, (self.nbrecord,))
        return cres.entry_point

    def test_shape(self):
        if False:
            i = 10
            return i + 15
        pyfunc = nested_array_shape
        cfunc = self.get_cfunc(pyfunc)
        self.assertEqual(pyfunc(self.a), cfunc(self.a))

    def test_strides(self):
        if False:
            print('Hello World!')
        pyfunc = nested_array_strides
        cfunc = self.get_cfunc(pyfunc)
        self.assertEqual(pyfunc(self.a), cfunc(self.a))

    def test_ndim(self):
        if False:
            print('Hello World!')
        pyfunc = nested_array_ndim
        cfunc = self.get_cfunc(pyfunc)
        self.assertEqual(pyfunc(self.a), cfunc(self.a))

    def test_nbytes(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = nested_array_nbytes
        cfunc = self.get_cfunc(pyfunc)
        self.assertEqual(pyfunc(self.a), cfunc(self.a))

    def test_size(self):
        if False:
            while True:
                i = 10
        pyfunc = nested_array_size
        cfunc = self.get_cfunc(pyfunc)
        self.assertEqual(pyfunc(self.a), cfunc(self.a))

    def test_itemsize(self):
        if False:
            while True:
                i = 10
        pyfunc = nested_array_itemsize
        cfunc = self.get_cfunc(pyfunc)
        self.assertEqual(pyfunc(self.a), cfunc(self.a))

class TestSlicedArrayAttr(MemoryLeakMixin, unittest.TestCase):

    def test_size_after_slicing(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = size_after_slicing_usecase
        cfunc = njit(pyfunc)
        arr = np.arange(2 * 5).reshape(2, 5)
        for i in range(arr.shape[0]):
            self.assertEqual(pyfunc(arr, i), cfunc(arr, i))
        arr = np.arange(2 * 5 * 3).reshape(2, 5, 3)
        for i in range(arr.shape[0]):
            self.assertEqual(pyfunc(arr, i), cfunc(arr, i))

class TestArrayCTypes(MemoryLeakMixin, TestCase):
    _numba_parallel_test_ = False

    def test_array_ctypes_data(self):
        if False:
            i = 10
            return i + 15
        pyfunc = array_ctypes_data
        cfunc = njit(pyfunc)
        arr = np.arange(3)
        self.assertEqual(pyfunc(arr), cfunc(arr))

    @skip_parfors_unsupported
    def test_array_ctypes_ref_error_in_parallel(self):
        if False:
            return 10
        from ctypes import CFUNCTYPE, c_void_p, c_int32, c_double, c_bool

        @CFUNCTYPE(c_bool, c_void_p, c_int32, c_void_p)
        def callback(inptr, size, outptr):
            if False:
                return 10
            try:
                inbuf = (c_double * size).from_address(inptr)
                outbuf = (c_double * 1).from_address(outptr)
                a = np.ndarray(size, buffer=inbuf, dtype=np.float64)
                b = np.ndarray(1, buffer=outbuf, dtype=np.float64)
                b[0] = (a + a.size)[0]
                return True
            except:
                import traceback
                traceback.print_exception()
                return False

        @njit(parallel=True)
        def foo(size):
            if False:
                print('Hello World!')
            arr = np.ones(size)
            out = np.empty(1)
            inct = arr.ctypes
            outct = out.ctypes
            status = callback(inct.data, size, outct.data)
            return (status, out[0])
        size = 3
        (status, got) = foo(size)
        self.assertTrue(status)
        self.assertPreciseEqual(got, (np.ones(size) + size)[0])

class TestRealImagAttr(MemoryLeakMixin, TestCase):

    def check_complex(self, pyfunc):
        if False:
            return 10
        cfunc = njit(pyfunc)
        size = 10
        arr = np.arange(size) + np.arange(size) * 10j
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
        arr = arr.reshape(2, 5)
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))

    def test_complex_real(self):
        if False:
            while True:
                i = 10
        self.check_complex(array_real)

    def test_complex_imag(self):
        if False:
            return 10
        self.check_complex(array_imag)

    def check_number_real(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = array_real
        cfunc = njit(pyfunc)
        size = 10
        arr = np.arange(size, dtype=dtype)
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
        arr = arr.reshape(2, 5)
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
        self.assertEqual(arr.data, pyfunc(arr).data)
        self.assertEqual(arr.data, cfunc(arr).data)
        real = cfunc(arr)
        self.assertNotEqual(arr[0, 0], 5)
        real[0, 0] = 5
        self.assertEqual(arr[0, 0], 5)

    def test_number_real(self):
        if False:
            print('Hello World!')
        '\n        Testing .real of non-complex dtypes\n        '
        for dtype in [np.uint8, np.int32, np.float32, np.float64]:
            self.check_number_real(dtype)

    def check_number_imag(self, dtype):
        if False:
            return 10
        pyfunc = array_imag
        cfunc = njit(pyfunc)
        size = 10
        arr = np.arange(size, dtype=dtype)
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
        arr = arr.reshape(2, 5)
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
        self.assertEqual(cfunc(arr).tolist(), np.zeros_like(arr).tolist())
        imag = cfunc(arr)
        with self.assertRaises(ValueError) as raises:
            imag[0] = 1
        self.assertEqual('assignment destination is read-only', str(raises.exception))

    def test_number_imag(self):
        if False:
            print('Hello World!')
        '\n        Testing .imag of non-complex dtypes\n        '
        for dtype in [np.uint8, np.int32, np.float32, np.float64]:
            self.check_number_imag(dtype)

    def test_record_real(self):
        if False:
            for i in range(10):
                print('nop')
        rectyp = np.dtype([('real', np.float32), ('imag', np.complex64)])
        arr = np.zeros(3, dtype=rectyp)
        arr['real'] = np.random.random(arr.size)
        arr['imag'] = np.random.random(arr.size) * 1.3j
        self.assertIs(array_real(arr), arr)
        self.assertEqual(array_imag(arr).tolist(), np.zeros_like(arr).tolist())
        jit_array_real = njit(array_real)
        jit_array_imag = njit(array_imag)
        with self.assertRaises(TypingError) as raises:
            jit_array_real(arr)
        self.assertIn('cannot access .real of array of Record', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            jit_array_imag(arr)
        self.assertIn('cannot access .imag of array of Record', str(raises.exception))

class TestJitclassFlagsSegfault(MemoryLeakMixin, TestCase):
    """Regression test for: https://github.com/numba/numba/issues/4775 """

    def test(self):
        if False:
            for i in range(10):
                print('nop')

        @jitclass(dict())
        class B(object):

            def __init__(self):
                if False:
                    print('Hello World!')
                pass

            def foo(self, X):
                if False:
                    while True:
                        i = 10
                X.flags
        Z = B()
        Z.foo(np.ones(4))
if __name__ == '__main__':
    unittest.main()