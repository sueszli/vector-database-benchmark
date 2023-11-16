import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
nrtjit = njit(_nrt=True, nogil=True)

def np_concatenate1(a, b, c):
    if False:
        return 10
    return np.concatenate((a, b, c))

def np_concatenate2(a, b, c, axis):
    if False:
        print('Hello World!')
    return np.concatenate((a, b, c), axis=axis)

def np_stack1(a, b, c):
    if False:
        i = 10
        return i + 15
    return np.stack((a, b, c))

def np_stack2(a, b, c, axis):
    if False:
        print('Hello World!')
    return np.stack((a, b, c), axis=axis)

def np_hstack(a, b, c):
    if False:
        print('Hello World!')
    return np.hstack((a, b, c))

def np_vstack(a, b, c):
    if False:
        while True:
            i = 10
    return np.vstack((a, b, c))

def np_row_stack(a, b, c):
    if False:
        print('Hello World!')
    return np.row_stack((a, b, c))

def np_dstack(a, b, c):
    if False:
        while True:
            i = 10
    return np.dstack((a, b, c))

def np_column_stack(a, b, c):
    if False:
        while True:
            i = 10
    return np.column_stack((a, b, c))

class BaseTest(TestCase):

    def check_outputs(self, pyfunc, argslist, exact=True):
        if False:
            for i in range(10):
                print('nop')
        cfunc = nrtjit(pyfunc)
        for args in argslist:
            expected = pyfunc(*args)
            ret = cfunc(*args)
            self.assertEqual(ret.size, expected.size)
            self.assertEqual(ret.dtype, expected.dtype)
            self.assertStridesEqual(ret, expected)
            if exact:
                np.testing.assert_equal(expected, ret)
            else:
                np.testing.assert_allclose(expected, ret)

class NrtRefCtTest(MemoryLeakMixin):

    def assert_array_nrt_refct(self, arr, expect):
        if False:
            return 10
        self.assertEqual(arr.base.refcount, expect)

class TestDynArray(NrtRefCtTest, TestCase):

    def test_empty_0d(self):
        if False:
            print('Hello World!')

        @nrtjit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            arr = np.empty(())
            arr[()] = 42
            return arr
        arr = foo()
        self.assert_array_nrt_refct(arr, 1)
        np.testing.assert_equal(42, arr)
        self.assertEqual(arr.size, 1)
        self.assertEqual(arr.shape, ())
        self.assertEqual(arr.dtype, np.dtype(np.float64))
        self.assertEqual(arr.strides, ())
        arr.fill(123)
        np.testing.assert_equal(123, arr)
        del arr

    def test_empty_1d(self):
        if False:
            i = 10
            return i + 15

        @nrtjit
        def foo(n):
            if False:
                print('Hello World!')
            arr = np.empty(n)
            for i in range(n):
                arr[i] = i
            return arr
        n = 3
        arr = foo(n)
        self.assert_array_nrt_refct(arr, 1)
        np.testing.assert_equal(np.arange(n), arr)
        self.assertEqual(arr.size, n)
        self.assertEqual(arr.shape, (n,))
        self.assertEqual(arr.dtype, np.dtype(np.float64))
        self.assertEqual(arr.strides, (np.dtype(np.float64).itemsize,))
        arr.fill(123)
        np.testing.assert_equal(123, arr)
        del arr

    def test_empty_2d(self):
        if False:
            return 10

        def pyfunc(m, n):
            if False:
                for i in range(10):
                    print('nop')
            arr = np.empty((m, n), np.int32)
            for i in range(m):
                for j in range(n):
                    arr[i, j] = i + j
            return arr
        cfunc = nrtjit(pyfunc)
        m = 4
        n = 3
        expected_arr = pyfunc(m, n)
        got_arr = cfunc(m, n)
        self.assert_array_nrt_refct(got_arr, 1)
        np.testing.assert_equal(expected_arr, got_arr)
        self.assertEqual(expected_arr.size, got_arr.size)
        self.assertEqual(expected_arr.shape, got_arr.shape)
        self.assertEqual(expected_arr.strides, got_arr.strides)
        del got_arr

    def test_empty_3d(self):
        if False:
            i = 10
            return i + 15

        def pyfunc(m, n, p):
            if False:
                for i in range(10):
                    print('nop')
            arr = np.empty((m, n, p), np.int32)
            for i in range(m):
                for j in range(n):
                    for k in range(p):
                        arr[i, j, k] = i + j + k
            return arr
        cfunc = nrtjit(pyfunc)
        m = 4
        n = 3
        p = 2
        expected_arr = pyfunc(m, n, p)
        got_arr = cfunc(m, n, p)
        self.assert_array_nrt_refct(got_arr, 1)
        np.testing.assert_equal(expected_arr, got_arr)
        self.assertEqual(expected_arr.size, got_arr.size)
        self.assertEqual(expected_arr.shape, got_arr.shape)
        self.assertEqual(expected_arr.strides, got_arr.strides)
        del got_arr

    def test_empty_2d_sliced(self):
        if False:
            for i in range(10):
                print('nop')

        def pyfunc(m, n, p):
            if False:
                for i in range(10):
                    print('nop')
            arr = np.empty((m, n), np.int32)
            for i in range(m):
                for j in range(n):
                    arr[i, j] = i + j
            return arr[p]
        cfunc = nrtjit(pyfunc)
        m = 4
        n = 3
        p = 2
        expected_arr = pyfunc(m, n, p)
        got_arr = cfunc(m, n, p)
        self.assert_array_nrt_refct(got_arr, 1)
        np.testing.assert_equal(expected_arr, got_arr)
        self.assertEqual(expected_arr.size, got_arr.size)
        self.assertEqual(expected_arr.shape, got_arr.shape)
        self.assertEqual(expected_arr.strides, got_arr.strides)
        del got_arr

    def test_return_global_array(self):
        if False:
            return 10
        y = np.ones(4, dtype=np.float32)
        initrefct = sys.getrefcount(y)

        def return_external_array():
            if False:
                print('Hello World!')
            return y
        cfunc = nrtjit(return_external_array)
        out = cfunc()
        self.assertEqual(initrefct + 1, sys.getrefcount(y))
        np.testing.assert_equal(y, out)
        np.testing.assert_equal(y, np.ones(4, dtype=np.float32))
        np.testing.assert_equal(out, np.ones(4, dtype=np.float32))
        del out
        gc.collect()
        self.assertEqual(initrefct + 1, sys.getrefcount(y))
        del cfunc
        gc.collect()
        self.assertEqual(initrefct, sys.getrefcount(y))

    def test_return_global_array_sliced(self):
        if False:
            print('Hello World!')
        y = np.ones(4, dtype=np.float32)

        def return_external_array():
            if False:
                print('Hello World!')
            return y[2:]
        cfunc = nrtjit(return_external_array)
        out = cfunc()
        self.assertIsNone(out.base)
        yy = y[2:]
        np.testing.assert_equal(yy, out)
        np.testing.assert_equal(yy, np.ones(2, dtype=np.float32))
        np.testing.assert_equal(out, np.ones(2, dtype=np.float32))

    def test_array_pass_through(self):
        if False:
            while True:
                i = 10

        def pyfunc(y):
            if False:
                print('Hello World!')
            return y
        arr = np.ones(4, dtype=np.float32)
        cfunc = nrtjit(pyfunc)
        expected = cfunc(arr)
        got = pyfunc(arr)
        np.testing.assert_equal(expected, arr)
        np.testing.assert_equal(expected, got)
        self.assertIs(expected, arr)
        self.assertIs(expected, got)

    def test_array_pass_through_sliced(self):
        if False:
            i = 10
            return i + 15

        def pyfunc(y):
            if False:
                for i in range(10):
                    print('nop')
            return y[y.size // 2:]
        arr = np.ones(4, dtype=np.float32)
        initrefct = sys.getrefcount(arr)
        cfunc = nrtjit(pyfunc)
        got = cfunc(arr)
        self.assertEqual(initrefct + 1, sys.getrefcount(arr))
        expected = pyfunc(arr)
        self.assertEqual(initrefct + 2, sys.getrefcount(arr))
        np.testing.assert_equal(expected, arr[arr.size // 2])
        np.testing.assert_equal(expected, got)
        del expected
        self.assertEqual(initrefct + 1, sys.getrefcount(arr))
        del got
        self.assertEqual(initrefct, sys.getrefcount(arr))

    def test_ufunc_with_allocated_output(self):
        if False:
            i = 10
            return i + 15

        def pyfunc(a, b):
            if False:
                for i in range(10):
                    print('nop')
            out = np.empty(a.shape)
            np.add(a, b, out)
            return out
        cfunc = nrtjit(pyfunc)
        arr_a = np.random.random(10)
        arr_b = np.random.random(10)
        np.testing.assert_equal(pyfunc(arr_a, arr_b), cfunc(arr_a, arr_b))
        self.assert_array_nrt_refct(cfunc(arr_a, arr_b), 1)
        arr_a = np.random.random(10).reshape(2, 5)
        arr_b = np.random.random(10).reshape(2, 5)
        np.testing.assert_equal(pyfunc(arr_a, arr_b), cfunc(arr_a, arr_b))
        self.assert_array_nrt_refct(cfunc(arr_a, arr_b), 1)
        arr_a = np.random.random(70).reshape(2, 5, 7)
        arr_b = np.random.random(70).reshape(2, 5, 7)
        np.testing.assert_equal(pyfunc(arr_a, arr_b), cfunc(arr_a, arr_b))
        self.assert_array_nrt_refct(cfunc(arr_a, arr_b), 1)

    def test_allocation_mt(self):
        if False:
            print('Hello World!')
        '\n        This test exercises the array allocation in multithreaded usecase.\n        This stress the freelist inside NRT.\n        '

        def pyfunc(inp):
            if False:
                print('Hello World!')
            out = np.empty(inp.size)
            for i in range(out.size):
                out[i] = 0
            for i in range(inp[0]):
                tmp = np.empty(inp.size)
                for j in range(tmp.size):
                    tmp[j] = inp[j]
                for j in range(tmp.size):
                    out[j] += tmp[j] + i
            return out
        cfunc = nrtjit(pyfunc)
        size = 10
        arr = np.random.randint(1, 10, size)
        frozen_arr = arr.copy()
        np.testing.assert_equal(pyfunc(arr), cfunc(arr))
        np.testing.assert_equal(frozen_arr, arr)
        workers = []
        inputs = []
        outputs = []

        def wrapped(inp, out):
            if False:
                i = 10
                return i + 15
            out[:] = cfunc(inp)
        for i in range(100):
            arr = np.random.randint(1, 10, size)
            out = np.empty_like(arr)
            thread = threading.Thread(target=wrapped, args=(arr, out), name='worker{0}'.format(i))
            workers.append(thread)
            inputs.append(arr)
            outputs.append(out)
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
        for (inp, out) in zip(inputs, outputs):
            np.testing.assert_equal(pyfunc(inp), out)

    def test_refct_mt(self):
        if False:
            while True:
                i = 10
        '\n        This test exercises the refct in multithreaded code\n        '

        def pyfunc(n, inp):
            if False:
                return 10
            out = np.empty(inp.size)
            for i in range(out.size):
                out[i] = inp[i] + 1
            for i in range(n):
                (out, inp) = (inp, out)
            return out
        cfunc = nrtjit(pyfunc)
        size = 10
        input = np.arange(size, dtype=float)
        expected_refct = sys.getrefcount(input)
        swapct = random.randrange(1000)
        expected = pyfunc(swapct, input)
        np.testing.assert_equal(expected, cfunc(swapct, input))
        del expected
        self.assertEqual(expected_refct, sys.getrefcount(input))
        workers = []
        outputs = []
        swapcts = []

        def wrapped(n, input, out):
            if False:
                return 10
            out[:] = cfunc(n, input)
        for i in range(100):
            out = np.empty(size)
            swapct = random.randrange(1000)
            thread = threading.Thread(target=wrapped, args=(swapct, input, out), name='worker{0}'.format(i))
            workers.append(thread)
            outputs.append(out)
            swapcts.append(swapct)
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
        for (swapct, out) in zip(swapcts, outputs):
            np.testing.assert_equal(pyfunc(swapct, input), out)
        del outputs, workers
        self.assertEqual(expected_refct, sys.getrefcount(input))

    @skip_if_32bit
    def test_invalid_size_array(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo(x):
            if False:
                while True:
                    i = 10
            np.empty(x)
        self.disable_leak_check()
        with self.assertRaises(MemoryError) as raises:
            foo(types.size_t.maxval // 8 // 2)
        self.assertIn('Allocation failed', str(raises.exception))

    def test_swap(self):
        if False:
            while True:
                i = 10

        def pyfunc(x, y, t):
            if False:
                i = 10
                return i + 15
            'Swap array x and y for t number of times\n            '
            for i in range(t):
                (x, y) = (y, x)
            return (x, y)
        cfunc = nrtjit(pyfunc)
        x = np.random.random(100)
        y = np.random.random(100)
        t = 100
        initrefct = (sys.getrefcount(x), sys.getrefcount(y))
        (expect, got) = (pyfunc(x, y, t), cfunc(x, y, t))
        self.assertIsNone(got[0].base)
        self.assertIsNone(got[1].base)
        np.testing.assert_equal(expect, got)
        del expect, got
        self.assertEqual(initrefct, (sys.getrefcount(x), sys.getrefcount(y)))

    def test_return_tuple_of_array(self):
        if False:
            i = 10
            return i + 15

        def pyfunc(x):
            if False:
                for i in range(10):
                    print('nop')
            y = np.empty(x.size)
            for i in range(y.size):
                y[i] = x[i] + 1
            return (x, y)
        cfunc = nrtjit(pyfunc)
        x = np.random.random(5)
        initrefct = sys.getrefcount(x)
        (expected_x, expected_y) = pyfunc(x)
        (got_x, got_y) = cfunc(x)
        self.assertIs(x, expected_x)
        self.assertIs(x, got_x)
        np.testing.assert_equal(expected_x, got_x)
        np.testing.assert_equal(expected_y, got_y)
        del expected_x, got_x
        self.assertEqual(initrefct, sys.getrefcount(x))
        self.assertEqual(sys.getrefcount(expected_y), sys.getrefcount(got_y))

    def test_return_tuple_of_array_created(self):
        if False:
            return 10

        def pyfunc(x):
            if False:
                i = 10
                return i + 15
            y = np.empty(x.size)
            for i in range(y.size):
                y[i] = x[i] + 1
            out = (y, y)
            return out
        cfunc = nrtjit(pyfunc)
        x = np.random.random(5)
        (expected_x, expected_y) = pyfunc(x)
        (got_x, got_y) = cfunc(x)
        np.testing.assert_equal(expected_x, got_x)
        np.testing.assert_equal(expected_y, got_y)
        self.assertEqual(2, sys.getrefcount(got_y))
        self.assertEqual(2, sys.getrefcount(got_y))

    def test_issue_with_return_leak(self):
        if False:
            print('Hello World!')
        '\n        Dispatcher returns a new reference.\n        It need to workaround it for now.\n        '

        @nrtjit
        def inner(out):
            if False:
                i = 10
                return i + 15
            return out

        def pyfunc(x):
            if False:
                while True:
                    i = 10
            return inner(x)
        cfunc = nrtjit(pyfunc)
        arr = np.arange(10)
        old_refct = sys.getrefcount(arr)
        self.assertEqual(old_refct, sys.getrefcount(pyfunc(arr)))
        self.assertEqual(old_refct, sys.getrefcount(cfunc(arr)))
        self.assertEqual(old_refct, sys.getrefcount(arr))

class ConstructorBaseTest(NrtRefCtTest):

    def check_0d(self, pyfunc):
        if False:
            i = 10
            return i + 15
        cfunc = nrtjit(pyfunc)
        expected = pyfunc()
        ret = cfunc()
        self.assert_array_nrt_refct(ret, 1)
        self.assertEqual(ret.size, expected.size)
        self.assertEqual(ret.shape, expected.shape)
        self.assertEqual(ret.dtype, expected.dtype)
        self.assertEqual(ret.strides, expected.strides)
        self.check_result_value(ret, expected)
        expected = np.empty_like(ret)
        expected.fill(123)
        ret.fill(123)
        np.testing.assert_equal(ret, expected)

    def check_1d(self, pyfunc):
        if False:
            while True:
                i = 10
        cfunc = nrtjit(pyfunc)
        n = 3
        expected = pyfunc(n)
        ret = cfunc(n)
        self.assert_array_nrt_refct(ret, 1)
        self.assertEqual(ret.size, expected.size)
        self.assertEqual(ret.shape, expected.shape)
        self.assertEqual(ret.dtype, expected.dtype)
        self.assertEqual(ret.strides, expected.strides)
        self.check_result_value(ret, expected)
        expected = np.empty_like(ret)
        expected.fill(123)
        ret.fill(123)
        np.testing.assert_equal(ret, expected)
        with self.assertRaises(ValueError) as cm:
            cfunc(-1)
        self.assertEqual(str(cm.exception), 'negative dimensions not allowed')

    def check_2d(self, pyfunc):
        if False:
            i = 10
            return i + 15
        cfunc = nrtjit(pyfunc)
        (m, n) = (2, 3)
        expected = pyfunc(m, n)
        ret = cfunc(m, n)
        self.assert_array_nrt_refct(ret, 1)
        self.assertEqual(ret.size, expected.size)
        self.assertEqual(ret.shape, expected.shape)
        self.assertEqual(ret.dtype, expected.dtype)
        self.assertEqual(ret.strides, expected.strides)
        self.check_result_value(ret, expected)
        expected = np.empty_like(ret)
        expected.fill(123)
        ret.fill(123)
        np.testing.assert_equal(ret, expected)
        with self.assertRaises(ValueError) as cm:
            cfunc(2, -1)
        self.assertEqual(str(cm.exception), 'negative dimensions not allowed')

    def check_alloc_size(self, pyfunc):
        if False:
            i = 10
            return i + 15
        'Checks that pyfunc will error, not segfaulting due to array size.'
        cfunc = nrtjit(pyfunc)
        with self.assertRaises(ValueError) as e:
            cfunc()
        self.assertIn('array is too big', str(e.exception))

class TestNdZeros(ConstructorBaseTest, TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestNdZeros, self).setUp()
        self.pyfunc = np.zeros

    def check_result_value(self, ret, expected):
        if False:
            i = 10
            return i + 15
        np.testing.assert_equal(ret, expected)

    def test_0d(self):
        if False:
            print('Hello World!')
        pyfunc = self.pyfunc

        def func():
            if False:
                return 10
            return pyfunc(())
        self.check_0d(func)

    def test_1d(self):
        if False:
            i = 10
            return i + 15
        pyfunc = self.pyfunc

        def func(n):
            if False:
                i = 10
                return i + 15
            return pyfunc(n)
        self.check_1d(func)

    def test_1d_dtype(self):
        if False:
            i = 10
            return i + 15
        pyfunc = self.pyfunc

        def func(n):
            if False:
                print('Hello World!')
            return pyfunc(n, np.int32)
        self.check_1d(func)

    def test_1d_dtype_instance(self):
        if False:
            i = 10
            return i + 15
        pyfunc = self.pyfunc
        _dtype = np.dtype('int32')

        def func(n):
            if False:
                while True:
                    i = 10
            return pyfunc(n, _dtype)
        self.check_1d(func)

    def test_1d_dtype_str(self):
        if False:
            print('Hello World!')
        pyfunc = self.pyfunc
        _dtype = 'int32'

        def func(n):
            if False:
                print('Hello World!')
            return pyfunc(n, _dtype)
        self.check_1d(func)

        def func(n):
            if False:
                print('Hello World!')
            return pyfunc(n, 'complex128')
        self.check_1d(func)

    def test_1d_dtype_str_alternative_spelling(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = self.pyfunc
        _dtype = 'i4'

        def func(n):
            if False:
                for i in range(10):
                    print('nop')
            return pyfunc(n, _dtype)
        self.check_1d(func)

        def func(n):
            if False:
                for i in range(10):
                    print('nop')
            return pyfunc(n, 'c8')
        self.check_1d(func)

    def test_1d_dtype_str_structured_dtype(self):
        if False:
            while True:
                i = 10
        pyfunc = self.pyfunc
        _dtype = 'i4, (2,3)f8'

        def func(n):
            if False:
                return 10
            return pyfunc(n, _dtype)
        self.check_1d(func)

    def test_1d_dtype_non_const_str(self):
        if False:
            while True:
                i = 10
        pyfunc = self.pyfunc

        @njit
        def func(n, dt):
            if False:
                i = 10
                return i + 15
            return pyfunc(n, dt)
        with self.assertRaises(TypingError) as raises:
            func(5, 'int32')
        excstr = str(raises.exception)
        msg = f'If np.{self.pyfunc.__name__} dtype is a string it must be a string constant.'
        self.assertIn(msg, excstr)

    def test_1d_dtype_invalid_str(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = self.pyfunc

        @njit
        def func(n):
            if False:
                return 10
            return pyfunc(n, 'ABCDEF')
        with self.assertRaises(TypingError) as raises:
            func(5)
        excstr = str(raises.exception)
        self.assertIn("Invalid NumPy dtype specified: 'ABCDEF'", excstr)

    def test_2d(self):
        if False:
            while True:
                i = 10
        pyfunc = self.pyfunc

        def func(m, n):
            if False:
                return 10
            return pyfunc((m, n))
        self.check_2d(func)

    def test_2d_shape_dtypes(self):
        if False:
            i = 10
            return i + 15
        pyfunc = self.pyfunc

        def func1(m, n):
            if False:
                return 10
            return pyfunc((np.int16(m), np.int32(n)))
        self.check_2d(func1)

        def func2(m, n):
            if False:
                i = 10
                return i + 15
            return pyfunc((np.int64(m), np.int8(n)))
        self.check_2d(func2)
        if config.IS_32BITS:
            cfunc = nrtjit(lambda m, n: pyfunc((m, n)))
            with self.assertRaises(ValueError):
                cfunc(np.int64(1 << 32 - 1), 1)

    def test_2d_dtype_kwarg(self):
        if False:
            i = 10
            return i + 15
        pyfunc = self.pyfunc

        def func(m, n):
            if False:
                while True:
                    i = 10
            return pyfunc((m, n), dtype=np.complex64)
        self.check_2d(func)

    def test_2d_dtype_str_kwarg(self):
        if False:
            return 10
        pyfunc = self.pyfunc

        def func(m, n):
            if False:
                while True:
                    i = 10
            return pyfunc((m, n), dtype='complex64')
        self.check_2d(func)

    def test_2d_dtype_str_kwarg_alternative_spelling(self):
        if False:
            print('Hello World!')
        pyfunc = self.pyfunc

        def func(m, n):
            if False:
                return 10
            return pyfunc((m, n), dtype='c8')
        self.check_2d(func)

    def test_alloc_size(self):
        if False:
            while True:
                i = 10
        pyfunc = self.pyfunc
        width = types.intp.bitwidth

        def gen_func(shape, dtype):
            if False:
                print('Hello World!')
            return lambda : pyfunc(shape, dtype)
        self.check_alloc_size(gen_func(1 << width - 2, np.intp))
        self.check_alloc_size(gen_func((1 << width - 8, 64), np.intp))

class TestNdOnes(TestNdZeros):

    def setUp(self):
        if False:
            return 10
        super(TestNdOnes, self).setUp()
        self.pyfunc = np.ones

    @unittest.expectedFailure
    def test_1d_dtype_str_structured_dtype(self):
        if False:
            while True:
                i = 10
        super().test_1d_dtype_str_structured_dtype()

class TestNdFull(ConstructorBaseTest, TestCase):

    def check_result_value(self, ret, expected):
        if False:
            for i in range(10):
                print('nop')
        np.testing.assert_equal(ret, expected)

    def test_0d(self):
        if False:
            return 10

        def func():
            if False:
                return 10
            return np.full((), 4.5)
        self.check_0d(func)

    def test_1d(self):
        if False:
            while True:
                i = 10

        def func(n):
            if False:
                print('Hello World!')
            return np.full(n, 4.5)
        self.check_1d(func)

    def test_1d_dtype(self):
        if False:
            print('Hello World!')

        def func(n):
            if False:
                return 10
            return np.full(n, 4.5, np.bool_)
        self.check_1d(func)

    def test_1d_dtype_instance(self):
        if False:
            for i in range(10):
                print('nop')
        dtype = np.dtype('bool')

        def func(n):
            if False:
                i = 10
                return i + 15
            return np.full(n, 4.5, dtype)
        self.check_1d(func)

    def test_1d_dtype_str(self):
        if False:
            print('Hello World!')

        def func(n):
            if False:
                i = 10
                return i + 15
            return np.full(n, 4.5, 'bool_')
        self.check_1d(func)

    def test_1d_dtype_str_alternative_spelling(self):
        if False:
            print('Hello World!')

        def func(n):
            if False:
                return 10
            return np.full(n, 4.5, '?')
        self.check_1d(func)

    def test_1d_dtype_non_const_str(self):
        if False:
            i = 10
            return i + 15

        @njit
        def func(n, fv, dt):
            if False:
                while True:
                    i = 10
            return np.full(n, fv, dt)
        with self.assertRaises(TypingError) as raises:
            func((5,), 4.5, 'int32')
        excstr = str(raises.exception)
        msg = 'If np.full dtype is a string it must be a string constant.'
        self.assertIn(msg, excstr)

    def test_1d_dtype_invalid_str(self):
        if False:
            i = 10
            return i + 15

        @njit
        def func(n, fv):
            if False:
                i = 10
                return i + 15
            return np.full(n, fv, 'ABCDEF')
        with self.assertRaises(TypingError) as raises:
            func((5,), 4.5)
        excstr = str(raises.exception)
        self.assertIn("Invalid NumPy dtype specified: 'ABCDEF'", excstr)

    def test_2d(self):
        if False:
            for i in range(10):
                print('nop')

        def func(m, n):
            if False:
                return 10
            return np.full((m, n), 4.5)
        self.check_2d(func)

    def test_2d_dtype_kwarg(self):
        if False:
            print('Hello World!')

        def func(m, n):
            if False:
                for i in range(10):
                    print('nop')
            return np.full((m, n), 1 + 4.5j, dtype=np.complex64)
        self.check_2d(func)

    def test_2d_dtype_from_type(self):
        if False:
            return 10

        def func(m, n):
            if False:
                for i in range(10):
                    print('nop')
            return np.full((m, n), np.int32(1))
        self.check_2d(func)

        def func(m, n):
            if False:
                i = 10
                return i + 15
            return np.full((m, n), np.complex128(1))
        self.check_2d(func)

        def func(m, n):
            if False:
                i = 10
                return i + 15
            return np.full((m, n), 1, dtype=np.int8)
        self.check_2d(func)

    def test_2d_shape_dtypes(self):
        if False:
            while True:
                i = 10

        def func1(m, n):
            if False:
                return 10
            return np.full((np.int16(m), np.int32(n)), 4.5)
        self.check_2d(func1)

        def func2(m, n):
            if False:
                return 10
            return np.full((np.int64(m), np.int8(n)), 4.5)
        self.check_2d(func2)
        if config.IS_32BITS:
            cfunc = nrtjit(lambda m, n: np.full((m, n), 4.5))
            with self.assertRaises(ValueError):
                cfunc(np.int64(1 << 32 - 1), 1)

    def test_alloc_size(self):
        if False:
            print('Hello World!')
        width = types.intp.bitwidth

        def gen_func(shape, value):
            if False:
                print('Hello World!')
            return lambda : np.full(shape, value)
        self.check_alloc_size(gen_func(1 << width - 2, 1))
        self.check_alloc_size(gen_func((1 << width - 8, 64), 1))

class ConstructorLikeBaseTest(object):

    def mutate_array(self, arr):
        if False:
            i = 10
            return i + 15
        try:
            arr.fill(42)
        except (TypeError, ValueError):
            fill_value = b'x' * arr.dtype.itemsize
            arr.fill(fill_value)

    def check_like(self, pyfunc, dtype):
        if False:
            while True:
                i = 10

        def check_arr(arr):
            if False:
                while True:
                    i = 10
            expected = pyfunc(arr)
            ret = cfunc(arr)
            self.assertEqual(ret.size, expected.size)
            self.assertEqual(ret.dtype, expected.dtype)
            self.assertStridesEqual(ret, expected)
            self.check_result_value(ret, expected)
            self.mutate_array(ret)
            self.mutate_array(expected)
            np.testing.assert_equal(ret, expected)
        orig = np.linspace(0, 5, 6).astype(dtype)
        cfunc = nrtjit(pyfunc)
        for shape in (6, (2, 3), (1, 2, 3), (3, 1, 2), ()):
            if shape == ():
                arr = orig[-1:].reshape(())
            else:
                arr = orig.reshape(shape)
            check_arr(arr)
            if arr.ndim > 0:
                check_arr(arr[::2])
            arr.flags['WRITEABLE'] = False
            with self.assertRaises(ValueError):
                arr[0] = 1
            check_arr(arr)
        check_arr(orig[0])

class TestNdEmptyLike(ConstructorLikeBaseTest, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestNdEmptyLike, self).setUp()
        self.pyfunc = np.empty_like

    def check_result_value(self, ret, expected):
        if False:
            i = 10
            return i + 15
        pass

    def test_like(self):
        if False:
            print('Hello World!')
        pyfunc = self.pyfunc

        def func(arr):
            if False:
                while True:
                    i = 10
            return pyfunc(arr)
        self.check_like(func, np.float64)

    def test_like_structured(self):
        if False:
            i = 10
            return i + 15
        dtype = np.dtype([('a', np.int16), ('b', np.float32)])
        pyfunc = self.pyfunc

        def func(arr):
            if False:
                while True:
                    i = 10
            return pyfunc(arr)
        self.check_like(func, dtype)

    def test_like_dtype(self):
        if False:
            return 10
        pyfunc = self.pyfunc

        def func(arr):
            if False:
                i = 10
                return i + 15
            return pyfunc(arr, np.int32)
        self.check_like(func, np.float64)

    def test_like_dtype_instance(self):
        if False:
            return 10
        dtype = np.dtype('int32')
        pyfunc = self.pyfunc

        def func(arr):
            if False:
                print('Hello World!')
            return pyfunc(arr, dtype)
        self.check_like(func, np.float64)

    def test_like_dtype_structured(self):
        if False:
            return 10
        dtype = np.dtype([('a', np.int16), ('b', np.float32)])
        pyfunc = self.pyfunc

        def func(arr):
            if False:
                i = 10
                return i + 15
            return pyfunc(arr, dtype)
        self.check_like(func, np.float64)

    def test_like_dtype_kwarg(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = self.pyfunc

        def func(arr):
            if False:
                for i in range(10):
                    print('nop')
            return pyfunc(arr, dtype=np.int32)
        self.check_like(func, np.float64)

    def test_like_dtype_str_kwarg(self):
        if False:
            while True:
                i = 10
        pyfunc = self.pyfunc

        def func(arr):
            if False:
                print('Hello World!')
            return pyfunc(arr, dtype='int32')
        self.check_like(func, np.float64)

    def test_like_dtype_str_kwarg_alternative_spelling(self):
        if False:
            while True:
                i = 10
        pyfunc = self.pyfunc

        def func(arr):
            if False:
                for i in range(10):
                    print('nop')
            return pyfunc(arr, dtype='i4')
        self.check_like(func, np.float64)

    def test_like_dtype_non_const_str(self):
        if False:
            while True:
                i = 10
        pyfunc = self.pyfunc

        @njit
        def func(n, dt):
            if False:
                return 10
            return pyfunc(n, dt)
        with self.assertRaises(TypingError) as raises:
            func(np.ones(4), 'int32')
        excstr = str(raises.exception)
        msg = f'If np.{self.pyfunc.__name__} dtype is a string it must be a string constant.'
        self.assertIn(msg, excstr)
        self.assertIn('{}(array(float64, 1d, C), unicode_type)'.format(pyfunc.__name__), excstr)

    def test_like_dtype_invalid_str(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = self.pyfunc

        @njit
        def func(n):
            if False:
                while True:
                    i = 10
            return pyfunc(n, 'ABCDEF')
        with self.assertRaises(TypingError) as raises:
            func(np.ones(4))
        excstr = str(raises.exception)
        self.assertIn("Invalid NumPy dtype specified: 'ABCDEF'", excstr)

class TestNdZerosLike(TestNdEmptyLike):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestNdZerosLike, self).setUp()
        self.pyfunc = np.zeros_like

    def check_result_value(self, ret, expected):
        if False:
            print('Hello World!')
        np.testing.assert_equal(ret, expected)

    def test_like_structured(self):
        if False:
            while True:
                i = 10
        super(TestNdZerosLike, self).test_like_structured()

    def test_like_dtype_structured(self):
        if False:
            i = 10
            return i + 15
        super(TestNdZerosLike, self).test_like_dtype_structured()

class TestNdOnesLike(TestNdZerosLike):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestNdOnesLike, self).setUp()
        self.pyfunc = np.ones_like
        self.expected_value = 1

    @unittest.expectedFailure
    def test_like_structured(self):
        if False:
            print('Hello World!')
        super(TestNdOnesLike, self).test_like_structured()

    @unittest.expectedFailure
    def test_like_dtype_structured(self):
        if False:
            while True:
                i = 10
        super(TestNdOnesLike, self).test_like_dtype_structured()

class TestNdFullLike(ConstructorLikeBaseTest, TestCase):

    def check_result_value(self, ret, expected):
        if False:
            return 10
        np.testing.assert_equal(ret, expected)

    def test_like(self):
        if False:
            while True:
                i = 10

        def func(arr):
            if False:
                print('Hello World!')
            return np.full_like(arr, 3.5)
        self.check_like(func, np.float64)

    @unittest.expectedFailure
    def test_like_structured(self):
        if False:
            return 10
        dtype = np.dtype([('a', np.int16), ('b', np.float32)])

        def func(arr):
            if False:
                i = 10
                return i + 15
            return np.full_like(arr, 4.5)
        self.check_like(func, dtype)

    def test_like_dtype(self):
        if False:
            while True:
                i = 10

        def func(arr):
            if False:
                return 10
            return np.full_like(arr, 4.5, np.bool_)
        self.check_like(func, np.float64)

    def test_like_dtype_instance(self):
        if False:
            while True:
                i = 10
        dtype = np.dtype('bool')

        def func(arr):
            if False:
                print('Hello World!')
            return np.full_like(arr, 4.5, dtype)
        self.check_like(func, np.float64)

    def test_like_dtype_kwarg(self):
        if False:
            for i in range(10):
                print('nop')

        def func(arr):
            if False:
                while True:
                    i = 10
            return np.full_like(arr, 4.5, dtype=np.bool_)
        self.check_like(func, np.float64)

    def test_like_dtype_str_kwarg(self):
        if False:
            return 10

        def func(arr):
            if False:
                while True:
                    i = 10
            return np.full_like(arr, 4.5, 'bool_')
        self.check_like(func, np.float64)

    def test_like_dtype_str_kwarg_alternative_spelling(self):
        if False:
            print('Hello World!')

        def func(arr):
            if False:
                i = 10
                return i + 15
            return np.full_like(arr, 4.5, dtype='?')
        self.check_like(func, np.float64)

    def test_like_dtype_non_const_str_kwarg(self):
        if False:
            return 10

        @njit
        def func(arr, fv, dt):
            if False:
                i = 10
                return i + 15
            return np.full_like(arr, fv, dt)
        with self.assertRaises(TypingError) as raises:
            func(np.ones(3), 4.5, 'int32')
        excstr = str(raises.exception)
        msg = 'If np.full_like dtype is a string it must be a string constant.'
        self.assertIn(msg, excstr)

    def test_like_dtype_invalid_str(self):
        if False:
            print('Hello World!')

        @njit
        def func(arr, fv):
            if False:
                for i in range(10):
                    print('nop')
            return np.full_like(arr, fv, 'ABCDEF')
        with self.assertRaises(TypingError) as raises:
            func(np.ones(4), 3.4)
        excstr = str(raises.exception)
        self.assertIn("Invalid NumPy dtype specified: 'ABCDEF'", excstr)

class TestNdIdentity(BaseTest):

    def check_identity(self, pyfunc):
        if False:
            while True:
                i = 10
        self.check_outputs(pyfunc, [(3,)])

    def test_identity(self):
        if False:
            for i in range(10):
                print('nop')

        def func(n):
            if False:
                return 10
            return np.identity(n)
        self.check_identity(func)

    def test_identity_dtype(self):
        if False:
            i = 10
            return i + 15
        for dtype in (np.complex64, np.int16, np.bool_, np.dtype('bool'), 'bool_'):

            def func(n):
                if False:
                    return 10
                return np.identity(n, dtype)
            self.check_identity(func)

    def test_like_dtype_non_const_str_kwarg(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def func(n, dt):
            if False:
                for i in range(10):
                    print('nop')
            return np.identity(n, dt)
        with self.assertRaises(TypingError) as raises:
            func(4, 'int32')
        excstr = str(raises.exception)
        msg = 'If np.identity dtype is a string it must be a string constant.'
        self.assertIn(msg, excstr)

class TestNdEye(BaseTest):

    def test_eye_n(self):
        if False:
            return 10

        def func(n):
            if False:
                while True:
                    i = 10
            return np.eye(n)
        self.check_outputs(func, [(1,), (3,)])

    def test_eye_n_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        for dt in (None, np.complex128, np.complex64(1)):

            def func(n, dtype=dt):
                if False:
                    return 10
                return np.eye(n, dtype=dtype)
            self.check_outputs(func, [(1,), (3,)])

    def test_eye_n_m(self):
        if False:
            while True:
                i = 10

        def func(n, m):
            if False:
                for i in range(10):
                    print('nop')
            return np.eye(n, m)
        self.check_outputs(func, [(1, 2), (3, 2), (0, 3)])

    def check_eye_n_m_k(self, func):
        if False:
            while True:
                i = 10
        self.check_outputs(func, [(1, 2, 0), (3, 4, 1), (3, 4, -1), (4, 3, -2), (4, 3, -5), (4, 3, 5)])

    def test_eye_n_m_k(self):
        if False:
            while True:
                i = 10

        def func(n, m, k):
            if False:
                return 10
            return np.eye(n, m, k)
        self.check_eye_n_m_k(func)

    def test_eye_n_m_k_dtype(self):
        if False:
            return 10

        def func(n, m, k):
            if False:
                for i in range(10):
                    print('nop')
            return np.eye(N=n, M=m, k=k, dtype=np.int16)
        self.check_eye_n_m_k(func)

    def test_eye_n_m_k_dtype_instance(self):
        if False:
            print('Hello World!')
        dtype = np.dtype('int16')

        def func(n, m, k):
            if False:
                while True:
                    i = 10
            return np.eye(N=n, M=m, k=k, dtype=dtype)
        self.check_eye_n_m_k(func)

class TestNdDiag(TestCase):

    def setUp(self):
        if False:
            return 10
        v = np.array([1, 2, 3])
        hv = np.array([[1, 2, 3]])
        vv = np.transpose(hv)
        self.vectors = [v, hv, vv]
        a3x4 = np.arange(12).reshape(3, 4)
        a4x3 = np.arange(12).reshape(4, 3)
        self.matricies = [a3x4, a4x3]

        def func(q):
            if False:
                i = 10
                return i + 15
            return np.diag(q)
        self.py = func
        self.jit = nrtjit(func)

        def func_kwarg(q, k=0):
            if False:
                while True:
                    i = 10
            return np.diag(q, k=k)
        self.py_kw = func_kwarg
        self.jit_kw = nrtjit(func_kwarg)

    def check_diag(self, pyfunc, nrtfunc, *args, **kwargs):
        if False:
            print('Hello World!')
        expected = pyfunc(*args, **kwargs)
        computed = nrtfunc(*args, **kwargs)
        self.assertEqual(computed.size, expected.size)
        self.assertEqual(computed.dtype, expected.dtype)
        np.testing.assert_equal(expected, computed)

    def test_diag_vect_create(self):
        if False:
            return 10
        for d in self.vectors:
            self.check_diag(self.py, self.jit, d)

    def test_diag_vect_create_kwarg(self):
        if False:
            i = 10
            return i + 15
        for k in range(-10, 10):
            for d in self.vectors:
                self.check_diag(self.py_kw, self.jit_kw, d, k=k)

    def test_diag_extract(self):
        if False:
            return 10
        for d in self.matricies:
            self.check_diag(self.py, self.jit, d)

    def test_diag_extract_kwarg(self):
        if False:
            print('Hello World!')
        for k in range(-4, 4):
            for d in self.matricies:
                self.check_diag(self.py_kw, self.jit_kw, d, k=k)

    def test_error_handling(self):
        if False:
            return 10
        d = np.array([[[1.0]]])
        cfunc = nrtjit(self.py)
        with self.assertRaises(TypeError):
            cfunc()
        with self.assertRaises(TypingError):
            cfunc(d)
        with self.assertRaises(TypingError):
            dfunc = nrtjit(self.py_kw)
            dfunc(d, k=3)

    def test_bad_shape(self):
        if False:
            i = 10
            return i + 15
        cfunc = nrtjit(self.py)
        msg = '.*The argument "v" must be array-like.*'
        with self.assertRaisesRegex(TypingError, msg) as raises:
            cfunc(None)

class TestLinspace(BaseTest):

    def test_linspace_2(self):
        if False:
            i = 10
            return i + 15

        def pyfunc(n, m):
            if False:
                return 10
            return np.linspace(n, m)
        self.check_outputs(pyfunc, [(0, 4), (1, 100), (-3.5, 2.5), (-3j, 2 + 3j), (2, 1), (1 + 0.5j, 1.5j)])

    def test_linspace_3(self):
        if False:
            print('Hello World!')

        def pyfunc(n, m, p):
            if False:
                print('Hello World!')
            return np.linspace(n, m, p)
        self.check_outputs(pyfunc, [(0, 4, 9), (1, 4, 3), (-3.5, 2.5, 8), (-3j, 2 + 3j, 7), (2, 1, 0), (1 + 0.5j, 1.5j, 5), (1, 1e+100, 1)])

    def test_linspace_accuracy(self):
        if False:
            i = 10
            return i + 15

        @nrtjit
        def foo(n, m, p):
            if False:
                while True:
                    i = 10
            return np.linspace(n, m, p)
        (n, m, p) = (0.0, 1.0, 100)
        self.assertPreciseEqual(foo(n, m, p), foo.py_func(n, m, p))

class TestNpyEmptyKeyword(TestCase):

    def _test_with_dtype_kw(self, dtype):
        if False:
            return 10

        def pyfunc(shape):
            if False:
                while True:
                    i = 10
            return np.empty(shape, dtype=dtype)
        shapes = [1, 5, 9]
        cfunc = nrtjit(pyfunc)
        for s in shapes:
            expected = pyfunc(s)
            got = cfunc(s)
            self.assertEqual(expected.dtype, got.dtype)
            self.assertEqual(expected.shape, got.shape)

    def test_with_dtype_kws(self):
        if False:
            print('Hello World!')
        for dtype in [np.int32, np.float32, np.complex64, np.dtype('complex64')]:
            self._test_with_dtype_kw(dtype)

    def _test_with_shape_and_dtype_kw(self, dtype):
        if False:
            i = 10
            return i + 15

        def pyfunc(shape):
            if False:
                while True:
                    i = 10
            return np.empty(shape=shape, dtype=dtype)
        shapes = [1, 5, 9]
        cfunc = nrtjit(pyfunc)
        for s in shapes:
            expected = pyfunc(s)
            got = cfunc(s)
            self.assertEqual(expected.dtype, got.dtype)
            self.assertEqual(expected.shape, got.shape)

    def test_with_shape_and_dtype_kws(self):
        if False:
            print('Hello World!')
        for dtype in [np.int32, np.float32, np.complex64, np.dtype('complex64')]:
            self._test_with_shape_and_dtype_kw(dtype)

    def test_empty_no_args(self):
        if False:
            i = 10
            return i + 15

        def pyfunc():
            if False:
                for i in range(10):
                    print('nop')
            return np.empty()
        cfunc = nrtjit(pyfunc)
        with self.assertRaises(TypingError):
            cfunc()

class TestNpArray(MemoryLeakMixin, BaseTest):

    def test_0d(self):
        if False:
            return 10

        def pyfunc(arg):
            if False:
                for i in range(10):
                    print('nop')
            return np.array(arg)
        cfunc = nrtjit(pyfunc)
        got = cfunc(42)
        self.assertPreciseEqual(got, np.array(42, dtype=np.intp))
        got = cfunc(2.5)
        self.assertPreciseEqual(got, np.array(2.5))

    def test_0d_with_dtype(self):
        if False:
            i = 10
            return i + 15

        def pyfunc(arg):
            if False:
                return 10
            return np.array(arg, dtype=np.int16)
        self.check_outputs(pyfunc, [(42,), (3.5,)])

    def test_1d(self):
        if False:
            while True:
                i = 10

        def pyfunc(arg):
            if False:
                i = 10
                return i + 15
            return np.array(arg)
        cfunc = nrtjit(pyfunc)
        got = cfunc([2, 3, 42])
        self.assertPreciseEqual(got, np.intp([2, 3, 42]))
        got = cfunc((1.0, 2.5j, 42))
        self.assertPreciseEqual(got, np.array([1.0, 2.5j, 42]))
        got = cfunc(())
        self.assertPreciseEqual(got, np.float64(()))

    def test_1d_with_dtype(self):
        if False:
            while True:
                i = 10

        def pyfunc(arg):
            if False:
                while True:
                    i = 10
            return np.array(arg, dtype=np.float32)
        self.check_outputs(pyfunc, [([2, 42],), ([3.5, 1.0],), ((1, 3.5, 42),), ((),)])

    def test_1d_with_str_dtype(self):
        if False:
            while True:
                i = 10

        def pyfunc(arg):
            if False:
                return 10
            return np.array(arg, dtype='float32')
        self.check_outputs(pyfunc, [([2, 42],), ([3.5, 1.0],), ((1, 3.5, 42),), ((),)])

    def test_1d_with_non_const_str_dtype(self):
        if False:
            print('Hello World!')

        @njit
        def func(arg, dt):
            if False:
                return 10
            return np.array(arg, dtype=dt)
        with self.assertRaises(TypingError) as raises:
            func((5, 3), 'int32')
        excstr = str(raises.exception)
        msg = f'If np.array dtype is a string it must be a string constant.'
        self.assertIn(msg, excstr)

    def test_2d(self):
        if False:
            return 10

        def pyfunc(arg):
            if False:
                i = 10
                return i + 15
            return np.array(arg)
        cfunc = nrtjit(pyfunc)
        got = cfunc([(1, 2), (3, 4)])
        self.assertPreciseEqual(got, np.intp([[1, 2], [3, 4]]))
        got = cfunc([(1, 2.5), (3, 4.5)])
        self.assertPreciseEqual(got, np.float64([[1, 2.5], [3, 4.5]]))
        got = cfunc(([1, 2], [3, 4]))
        self.assertPreciseEqual(got, np.intp([[1, 2], [3, 4]]))
        got = cfunc(([1, 2], [3.5, 4.5]))
        self.assertPreciseEqual(got, np.float64([[1, 2], [3.5, 4.5]]))
        got = cfunc(((1.5, 2), (3.5, 4.5)))
        self.assertPreciseEqual(got, np.float64([[1.5, 2], [3.5, 4.5]]))
        got = cfunc(((), ()))
        self.assertPreciseEqual(got, np.float64(((), ())))

    def test_2d_with_dtype(self):
        if False:
            for i in range(10):
                print('nop')

        def pyfunc(arg):
            if False:
                while True:
                    i = 10
            return np.array(arg, dtype=np.int32)
        cfunc = nrtjit(pyfunc)
        got = cfunc([(1, 2.5), (3, 4.5)])
        self.assertPreciseEqual(got, np.int32([[1, 2], [3, 4]]))

    def test_raises(self):
        if False:
            i = 10
            return i + 15

        def pyfunc(arg):
            if False:
                return 10
            return np.array(arg)
        cfunc = nrtjit(pyfunc)

        @contextlib.contextmanager
        def check_raises(msg):
            if False:
                i = 10
                return i + 15
            with self.assertRaises(TypingError) as raises:
                yield
            self.assertIn(msg, str(raises.exception))
        with check_raises('array(float64, 1d, C) not allowed in a homogeneous sequence'):
            cfunc(np.array([1.0]))
        with check_raises('type Tuple(int64, reflected list(int64)<iv=None>) does not have a regular shape'):
            cfunc((np.int64(1), [np.int64(2)]))
        with check_raises('cannot convert Tuple(int64, Record(a[type=int32;offset=0],b[type=float32;offset=4];8;False)) to a homogeneous type'):
            st = np.dtype([('a', 'i4'), ('b', 'f4')])
            val = np.zeros(1, dtype=st)[0]
            cfunc(((1, 2), (np.int64(1), val)))

    def test_bad_array(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def func(obj):
            if False:
                print('Hello World!')
            return np.array(obj)
        msg = '.*The argument "object" must be array-like.*'
        with self.assertRaisesRegex(TypingError, msg) as raises:
            func(None)

    def test_bad_dtype(self):
        if False:
            return 10

        @njit
        def func(obj, dt):
            if False:
                return 10
            return np.array(obj, dt)
        msg = '.*The argument "dtype" must be a data-type if it is provided.*'
        with self.assertRaisesRegex(TypingError, msg) as raises:
            func(5, 4)

class TestNpConcatenate(MemoryLeakMixin, TestCase):
    """
    Tests for np.concatenate().
    """

    def _3d_arrays(self):
        if False:
            print('Hello World!')
        a = np.arange(24).reshape((4, 3, 2))
        b = a + 10
        c = (b + 10).copy(order='F')
        d = (c + 10)[::-1]
        e = (d + 10)[..., ::-1]
        return (a, b, c, d, e)

    @contextlib.contextmanager
    def assert_invalid_sizes_over_dim(self, axis):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError) as raises:
            yield
        self.assertIn('input sizes over dimension %d do not match' % axis, str(raises.exception))

    def test_3d(self):
        if False:
            while True:
                i = 10
        pyfunc = np_concatenate2
        cfunc = nrtjit(pyfunc)

        def check(a, b, c, axis):
            if False:
                i = 10
                return i + 15
            for ax in (axis, -3 + axis):
                expected = pyfunc(a, b, c, axis=ax)
                got = cfunc(a, b, c, axis=ax)
                self.assertPreciseEqual(got, expected)

        def check_all_axes(a, b, c):
            if False:
                return 10
            for axis in range(3):
                check(a, b, c, axis)
        (a, b, c, d, e) = self._3d_arrays()
        check_all_axes(a, b, b)
        check_all_axes(a, b, c)
        check_all_axes(a.T, b.T, a.T)
        check_all_axes(a.T, b.T, c.T)
        check_all_axes(a.T, b.T, d.T)
        check_all_axes(d.T, e.T, d.T)
        check(a[1:], b, c[::-1], axis=0)
        check(a, b[:, 1:], c, axis=1)
        check(a, b, c[:, :, 1:], axis=2)
        check_all_axes(a, b.astype(np.float64), b)
        self.disable_leak_check()
        for axis in (1, 2, -2, -1):
            with self.assert_invalid_sizes_over_dim(0):
                cfunc(a[1:], b, b, axis)
        for axis in (0, 2, -3, -1):
            with self.assert_invalid_sizes_over_dim(1):
                cfunc(a, b[:, 1:], b, axis)

    def test_3d_no_axis(self):
        if False:
            i = 10
            return i + 15
        pyfunc = np_concatenate1
        cfunc = nrtjit(pyfunc)

        def check(a, b, c):
            if False:
                while True:
                    i = 10
            expected = pyfunc(a, b, c)
            got = cfunc(a, b, c)
            self.assertPreciseEqual(got, expected)
        (a, b, c, d, e) = self._3d_arrays()
        check(a, b, b)
        check(a, b, c)
        check(a.T, b.T, a.T)
        check(a.T, b.T, c.T)
        check(a.T, b.T, d.T)
        check(d.T, e.T, d.T)
        check(a[1:], b, c[::-1])
        self.disable_leak_check()
        with self.assert_invalid_sizes_over_dim(1):
            cfunc(a, b[:, 1:], b)

    def test_typing_errors(self):
        if False:
            return 10
        pyfunc = np_concatenate1
        cfunc = nrtjit(pyfunc)
        a = np.arange(15)
        b = a.reshape((3, 5))
        c = a.astype(np.dtype([('x', np.int8)]))
        d = np.array(42)
        with self.assertTypingError() as raises:
            cfunc(a, b, b)
        self.assertIn('all the input arrays must have same number of dimensions', str(raises.exception))
        with self.assertTypingError() as raises:
            cfunc(a, c, c)
        self.assertIn('input arrays must have compatible dtypes', str(raises.exception))
        with self.assertTypingError() as raises:
            cfunc(d, d, d)
        self.assertIn('zero-dimensional arrays cannot be concatenated', str(raises.exception))
        with self.assertTypingError() as raises:
            cfunc(c, 1, c)
        self.assertIn('expecting a non-empty tuple of arrays', str(raises.exception))

@unittest.skipUnless(hasattr(np, 'stack'), "this Numpy doesn't have np.stack()")
class TestNpStack(MemoryLeakMixin, TestCase):
    """
    Tests for np.stack().
    """

    def _3d_arrays(self):
        if False:
            i = 10
            return i + 15
        a = np.arange(24).reshape((4, 3, 2))
        b = a + 10
        c = (b + 10).copy(order='F')
        d = (c + 10)[::-1]
        e = (d + 10)[..., ::-1]
        return (a, b, c, d, e)

    @contextlib.contextmanager
    def assert_invalid_sizes(self):
        if False:
            return 10
        with self.assertRaises(ValueError) as raises:
            yield
        self.assertIn('all input arrays must have the same shape', str(raises.exception))

    def check_stack(self, pyfunc, cfunc, args):
        if False:
            i = 10
            return i + 15
        expected = pyfunc(*args)
        got = cfunc(*args)
        self.assertEqual(got.shape, expected.shape)
        self.assertPreciseEqual(got.flatten(), expected.flatten())

    def check_3d(self, pyfunc, cfunc, generate_starargs):
        if False:
            for i in range(10):
                print('nop')

        def check(a, b, c, args):
            if False:
                return 10
            self.check_stack(pyfunc, cfunc, (a, b, c) + args)

        def check_all_axes(a, b, c):
            if False:
                i = 10
                return i + 15
            for args in generate_starargs():
                check(a, b, c, args)
        (a, b, c, d, e) = self._3d_arrays()
        check_all_axes(a, b, b)
        check_all_axes(a, b, c)
        check_all_axes(a.T, b.T, a.T)
        check_all_axes(a.T, b.T, c.T)
        check_all_axes(a.T, b.T, d.T)
        check_all_axes(d.T, e.T, d.T)
        check_all_axes(a, b.astype(np.float64), b)

    def check_runtime_errors(self, cfunc, generate_starargs):
        if False:
            return 10
        self.assert_no_memory_leak()
        self.disable_leak_check()
        (a, b, c, d, e) = self._3d_arrays()
        with self.assert_invalid_sizes():
            args = next(generate_starargs())
            cfunc(a[:-1], b, c, *args)

    def test_3d(self):
        if False:
            return 10
        '\n        stack(3d arrays, axis)\n        '
        pyfunc = np_stack2
        cfunc = nrtjit(pyfunc)

        def generate_starargs():
            if False:
                i = 10
                return i + 15
            for axis in range(3):
                yield (axis,)
                yield (-3 + axis,)
        self.check_3d(pyfunc, cfunc, generate_starargs)
        self.check_runtime_errors(cfunc, generate_starargs)

    def test_3d_no_axis(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        stack(3d arrays)\n        '
        pyfunc = np_stack1
        cfunc = nrtjit(pyfunc)

        def generate_starargs():
            if False:
                i = 10
                return i + 15
            yield ()
        self.check_3d(pyfunc, cfunc, generate_starargs)
        self.check_runtime_errors(cfunc, generate_starargs)

    def test_0d(self):
        if False:
            return 10
        '\n        stack(0d arrays)\n        '
        pyfunc = np_stack1
        cfunc = nrtjit(pyfunc)
        a = np.array(42)
        b = np.array(-5j)
        c = np.array(True)
        self.check_stack(pyfunc, cfunc, (a, b, c))

    def check_xxstack(self, pyfunc, cfunc):
        if False:
            for i in range(10):
                print('nop')
        '\n        3d and 0d tests for hstack(), vstack(), dstack().\n        '

        def generate_starargs():
            if False:
                print('Hello World!')
            yield ()
        self.check_3d(pyfunc, cfunc, generate_starargs)
        a = np.array(42)
        b = np.array(-5j)
        c = np.array(True)
        self.check_stack(pyfunc, cfunc, (a, b, a))

    def test_hstack(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = np_hstack
        cfunc = nrtjit(pyfunc)
        self.check_xxstack(pyfunc, cfunc)
        a = np.arange(5)
        b = np.arange(6) + 10
        self.check_stack(pyfunc, cfunc, (a, b, b))
        a = np.arange(6).reshape((2, 3))
        b = np.arange(8).reshape((2, 4)) + 100
        self.check_stack(pyfunc, cfunc, (a, b, a))

    def test_vstack(self):
        if False:
            for i in range(10):
                print('nop')
        for pyfunc in (np_vstack, np_row_stack):
            cfunc = nrtjit(pyfunc)
            self.check_xxstack(pyfunc, cfunc)
            a = np.arange(5)
            b = a + 10
            self.check_stack(pyfunc, cfunc, (a, b, b))
            a = np.arange(6).reshape((3, 2))
            b = np.arange(8).reshape((4, 2)) + 100
            self.check_stack(pyfunc, cfunc, (a, b, b))

    def test_dstack(self):
        if False:
            while True:
                i = 10
        pyfunc = np_dstack
        cfunc = nrtjit(pyfunc)
        self.check_xxstack(pyfunc, cfunc)
        a = np.arange(5)
        b = a + 10
        self.check_stack(pyfunc, cfunc, (a, b, b))
        a = np.arange(12).reshape((3, 4))
        b = a + 100
        self.check_stack(pyfunc, cfunc, (a, b, b))

    def test_column_stack(self):
        if False:
            print('Hello World!')
        pyfunc = np_column_stack
        cfunc = nrtjit(pyfunc)
        a = np.arange(4)
        b = a + 10
        c = np.arange(12).reshape((4, 3))
        self.check_stack(pyfunc, cfunc, (a, b, c))
        self.assert_no_memory_leak()
        self.disable_leak_check()
        a = np.array(42)
        with self.assertTypingError():
            cfunc((a, a, a))
        a = a.reshape((1, 1, 1))
        with self.assertTypingError():
            cfunc((a, a, a))

    def test_bad_arrays(self):
        if False:
            print('Hello World!')
        for pyfunc in (np_stack1, np_hstack, np_vstack, np_dstack, np_column_stack):
            cfunc = nrtjit(pyfunc)
            c = np.arange(12).reshape((4, 3))
            with self.assertTypingError() as raises:
                cfunc(c, 1, c)
            self.assertIn('expecting a non-empty tuple of arrays', str(raises.exception))

def benchmark_refct_speed():
    if False:
        return 10

    def pyfunc(x, y, t):
        if False:
            i = 10
            return i + 15
        'Swap array x and y for t number of times\n        '
        for i in range(t):
            (x, y) = (y, x)
        return (x, y)
    cfunc = nrtjit(pyfunc)
    x = np.random.random(100)
    y = np.random.random(100)
    t = 10000

    def bench_pyfunc():
        if False:
            return 10
        pyfunc(x, y, t)

    def bench_cfunc():
        if False:
            i = 10
            return i + 15
        cfunc(x, y, t)
    python_time = utils.benchmark(bench_pyfunc)
    numba_time = utils.benchmark(bench_cfunc)
    print(python_time)
    print(numba_time)
if __name__ == '__main__':
    unittest.main()