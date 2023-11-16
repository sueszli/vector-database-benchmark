import itertools
import numpy as np
from numba import jit, typeof
from numba.core import types
from numba.core.compiler import compile_isolated
from numba.tests.support import TestCase, CompilationCache, MemoryLeakMixin, tag
import unittest

def array_iter(arr):
    if False:
        while True:
            i = 10
    total = 0
    for (i, v) in enumerate(arr):
        total += i * v
    return total

def array_iter_items(arr):
    if False:
        for i in range(10):
            print('nop')
    return list(iter(arr))

def array_view_iter(arr, idx):
    if False:
        while True:
            i = 10
    total = 0
    for (i, v) in enumerate(arr[idx]):
        total += i * v
    return total

def array_flat(arr, out):
    if False:
        return 10
    for (i, v) in enumerate(arr.flat):
        out[i] = v

def array_flat_getitem(arr, ind):
    if False:
        print('Hello World!')
    return arr.flat[ind]

def array_flat_setitem(arr, ind, val):
    if False:
        i = 10
        return i + 15
    arr.flat[ind] = val

def array_flat_sum(arr):
    if False:
        while True:
            i = 10
    s = 0
    for (i, v) in enumerate(arr.flat):
        s = s + (i + 1) * v
    return s

def array_flat_len(arr):
    if False:
        for i in range(10):
            print('nop')
    return len(arr.flat)

def array_ndenumerate_sum(arr):
    if False:
        print('Hello World!')
    s = 0
    for ((i, j), v) in np.ndenumerate(arr):
        s = s + (i + 1) * (j + 1) * v
    return s

def np_ndindex_empty():
    if False:
        for i in range(10):
            print('nop')
    s = 0
    for ind in np.ndindex(()):
        s += s + len(ind) + 1
    return s

def np_ndindex(x, y):
    if False:
        i = 10
        return i + 15
    s = 0
    n = 0
    for (i, j) in np.ndindex(x, y):
        s = s + (i + 1) * (j + 1)
    return s

def np_ndindex_array(arr):
    if False:
        while True:
            i = 10
    s = 0
    n = 0
    for indices in np.ndindex(arr.shape):
        for (i, j) in enumerate(indices):
            s = s + (i + 1) * (j + 1)
    return s

def np_nditer1(a):
    if False:
        for i in range(10):
            print('nop')
    res = []
    for u in np.nditer(a):
        res.append(u.item())
    return res

def np_nditer2(a, b):
    if False:
        while True:
            i = 10
    res = []
    for (u, v) in np.nditer((a, b)):
        res.append((u.item(), v.item()))
    return res

def np_nditer3(a, b, c):
    if False:
        while True:
            i = 10
    res = []
    for (u, v, w) in np.nditer((a, b, c)):
        res.append((u.item(), v.item(), w.item()))
    return res

def iter_next(arr):
    if False:
        for i in range(10):
            print('nop')
    it = iter(arr)
    it2 = iter(arr)
    return (next(it), next(it), next(it2))

def array_flat_premature_free(size):
    if False:
        i = 10
        return i + 15
    x = np.arange(size)
    res = np.zeros_like(x, dtype=np.intp)
    for (i, v) in enumerate(x.flat):
        res[i] = v
    return res

def array_ndenumerate_premature_free(size):
    if False:
        i = 10
        return i + 15
    x = np.arange(size)
    res = np.zeros_like(x, dtype=np.intp)
    for (i, v) in np.ndenumerate(x):
        res[i] = v
    return res

class TestArrayIterators(MemoryLeakMixin, TestCase):
    """
    Test array.flat, np.ndenumerate(), etc.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestArrayIterators, self).setUp()
        self.ccache = CompilationCache()

    def check_array_iter_1d(self, arr):
        if False:
            print('Hello World!')
        pyfunc = array_iter
        cres = compile_isolated(pyfunc, [typeof(arr)])
        cfunc = cres.entry_point
        expected = pyfunc(arr)
        self.assertPreciseEqual(cfunc(arr), expected)

    def check_array_iter_items(self, arr):
        if False:
            while True:
                i = 10
        pyfunc = array_iter_items
        cres = compile_isolated(pyfunc, [typeof(arr)])
        cfunc = cres.entry_point
        expected = pyfunc(arr)
        self.assertPreciseEqual(cfunc(arr), expected)

    def check_array_view_iter(self, arr, index):
        if False:
            i = 10
            return i + 15
        pyfunc = array_view_iter
        cres = compile_isolated(pyfunc, [typeof(arr), typeof(index)])
        cfunc = cres.entry_point
        expected = pyfunc(arr, index)
        self.assertPreciseEqual(cfunc(arr, index), expected)

    def check_array_flat(self, arr, arrty=None):
        if False:
            while True:
                i = 10
        out = np.zeros(arr.size, dtype=arr.dtype)
        nb_out = out.copy()
        if arrty is None:
            arrty = typeof(arr)
        cres = compile_isolated(array_flat, [arrty, typeof(out)])
        cfunc = cres.entry_point
        array_flat(arr, out)
        cfunc(arr, nb_out)
        self.assertPreciseEqual(out, nb_out)

    def check_array_unary(self, arr, arrty, func):
        if False:
            return 10
        cres = compile_isolated(func, [arrty])
        cfunc = cres.entry_point
        self.assertPreciseEqual(cfunc(arr), func(arr))

    def check_array_flat_sum(self, arr, arrty):
        if False:
            return 10
        self.check_array_unary(arr, arrty, array_flat_sum)

    def check_array_ndenumerate_sum(self, arr, arrty):
        if False:
            for i in range(10):
                print('nop')
        self.check_array_unary(arr, arrty, array_ndenumerate_sum)

    def test_array_iter(self):
        if False:
            return 10
        arr = np.arange(6)
        self.check_array_iter_1d(arr)
        self.check_array_iter_items(arr)
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        self.check_array_iter_1d(arr)
        self.check_array_iter_items(arr)
        arr = np.bool_([1, 0, 0, 1])
        self.check_array_iter_1d(arr)
        self.check_array_iter_items(arr)
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.check_array_iter_items(arr)
        self.check_array_iter_items(arr.T)

    def test_array_iter_yielded_order(self):
        if False:
            for i in range(10):
                print('nop')

        @jit(nopython=True)
        def foo(arr):
            if False:
                print('Hello World!')
            t = []
            for y1 in arr:
                for y2 in y1:
                    t.append(y2.ravel())
            return t
        arr = np.arange(24).reshape((2, 3, 4), order='F')
        expected = foo.py_func(arr)
        got = foo(arr)
        self.assertPreciseEqual(expected, got)
        arr = np.arange(64).reshape((4, 8, 2), order='F')[::2, :, :]
        expected = foo.py_func(arr)
        got = foo(arr)
        self.assertPreciseEqual(expected, got)
        arr = np.arange(64).reshape((4, 8, 2), order='F')[:, ::2, :]
        expected = foo.py_func(arr)
        got = foo(arr)
        self.assertPreciseEqual(expected, got)
        arr = np.arange(64).reshape((4, 8, 2), order='F')[:, :, ::2]
        expected = foo.py_func(arr)
        got = foo(arr)
        self.assertPreciseEqual(expected, got)

        @jit(nopython=True)
        def flag_check(arr):
            if False:
                return 10
            out = []
            for sub in arr:
                out.append((sub, sub.flags.c_contiguous, sub.flags.f_contiguous))
            return out
        arr = np.arange(10).reshape((2, 5), order='F')
        expected = flag_check.py_func(arr)
        got = flag_check(arr)
        self.assertEqual(len(expected), len(got))
        (ex_arr, e_flag_c, e_flag_f) = expected[0]
        (go_arr, g_flag_c, g_flag_f) = got[0]
        np.testing.assert_allclose(ex_arr, go_arr)
        self.assertEqual(e_flag_c, g_flag_c)
        self.assertEqual(e_flag_f, g_flag_f)

    def test_array_view_iter(self):
        if False:
            for i in range(10):
                print('nop')
        arr = np.arange(12).reshape((3, 4))
        self.check_array_view_iter(arr, 1)
        self.check_array_view_iter(arr.T, 1)
        arr = arr[::2]
        self.check_array_view_iter(arr, 1)
        arr = np.bool_([1, 0, 0, 1]).reshape((2, 2))
        self.check_array_view_iter(arr, 1)

    def test_array_flat_3d(self):
        if False:
            while True:
                i = 10
        arr = np.arange(24).reshape(4, 2, 3)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 3)
        self.assertEqual(arrty.layout, 'C')
        self.assertTrue(arr.flags.c_contiguous)
        self.check_array_flat(arr)
        arr = arr.transpose()
        self.assertFalse(arr.flags.c_contiguous)
        self.assertTrue(arr.flags.f_contiguous)
        self.assertEqual(typeof(arr).layout, 'F')
        self.check_array_flat(arr)
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        self.assertEqual(typeof(arr).layout, 'A')
        self.check_array_flat(arr)
        arr = np.bool_([1, 0, 0, 1] * 2).reshape((2, 2, 2))
        self.check_array_flat(arr)

    def test_array_flat_empty(self):
        if False:
            i = 10
            return i + 15
        arr = np.zeros(0, dtype=np.int32)
        arr = arr.reshape(0, 2)
        arrty = types.Array(types.int32, 2, layout='C')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        self.check_array_flat_sum(arr, arrty)
        arr = arr.reshape(2, 0)
        arrty = types.Array(types.int32, 2, layout='C')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        self.check_array_flat_sum(arr, arrty)

    def test_array_flat_getitem(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = array_flat_getitem

        def check(arr, ind):
            if False:
                while True:
                    i = 10
            cr = self.ccache.compile(pyfunc, (typeof(arr), typeof(ind)))
            expected = pyfunc(arr, ind)
            self.assertEqual(cr.entry_point(arr, ind), expected)
        arr = np.arange(24).reshape(4, 2, 3)
        for i in range(arr.size):
            check(arr, i)
        arr = arr.T
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(arr, i)
        arr = np.array([42]).reshape(())
        for i in range(arr.size):
            check(arr, i)
        arr = np.bool_([1, 0, 0, 1])
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(arr, i)

    def test_array_flat_setitem(self):
        if False:
            i = 10
            return i + 15
        pyfunc = array_flat_setitem

        def check(arr, ind):
            if False:
                return 10
            arrty = typeof(arr)
            cr = self.ccache.compile(pyfunc, (arrty, typeof(ind), arrty.dtype))
            expected = np.copy(arr)
            got = np.copy(arr)
            pyfunc(expected, ind, 123)
            cr.entry_point(got, ind, 123)
            self.assertPreciseEqual(got, expected)
        arr = np.arange(24).reshape(4, 2, 3)
        for i in range(arr.size):
            check(arr, i)
        arr = arr.T
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(arr, i)
        arr = np.array([42]).reshape(())
        for i in range(arr.size):
            check(arr, i)
        arr = np.bool_([1, 0, 0, 1])
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(arr, i)

    def test_array_flat_len(self):
        if False:
            i = 10
            return i + 15
        pyfunc = array_flat_len

        def check(arr):
            if False:
                i = 10
                return i + 15
            cr = self.ccache.compile(pyfunc, (typeof(arr),))
            expected = pyfunc(arr)
            self.assertPreciseEqual(cr.entry_point(arr), expected)
        arr = np.arange(24).reshape(4, 2, 3)
        check(arr)
        arr = arr.T
        check(arr)
        arr = arr[::2]
        check(arr)
        arr = np.array([42]).reshape(())
        check(arr)

    def test_array_flat_premature_free(self):
        if False:
            for i in range(10):
                print('nop')
        cres = compile_isolated(array_flat_premature_free, [types.intp])
        cfunc = cres.entry_point
        expect = array_flat_premature_free(6)
        got = cfunc(6)
        self.assertTrue(got.sum())
        self.assertPreciseEqual(expect, got)

    def test_array_ndenumerate_2d(self):
        if False:
            while True:
                i = 10
        arr = np.arange(12).reshape(4, 3)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 2)
        self.assertEqual(arrty.layout, 'C')
        self.assertTrue(arr.flags.c_contiguous)
        self.check_array_ndenumerate_sum(arr, arrty)
        arr = arr.transpose()
        self.assertFalse(arr.flags.c_contiguous)
        self.assertTrue(arr.flags.f_contiguous)
        arrty = typeof(arr)
        self.assertEqual(arrty.layout, 'F')
        self.check_array_ndenumerate_sum(arr, arrty)
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        arrty = typeof(arr)
        self.assertEqual(arrty.layout, 'A')
        self.check_array_ndenumerate_sum(arr, arrty)
        arr = np.bool_([1, 0, 0, 1]).reshape((2, 2))
        self.check_array_ndenumerate_sum(arr, typeof(arr))

    def test_array_ndenumerate_empty(self):
        if False:
            for i in range(10):
                print('nop')
        arr = np.zeros(0, dtype=np.int32)
        arr = arr.reshape(0, 2)
        arrty = types.Array(types.int32, 2, layout='C')
        self.check_array_ndenumerate_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        self.check_array_ndenumerate_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        self.check_array_ndenumerate_sum(arr, arrty)
        arr = arr.reshape(2, 0)
        arrty = types.Array(types.int32, 2, layout='C')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        self.check_array_flat_sum(arr, arrty)

    def test_array_ndenumerate_premature_free(self):
        if False:
            while True:
                i = 10
        cres = compile_isolated(array_ndenumerate_premature_free, [types.intp])
        cfunc = cres.entry_point
        expect = array_ndenumerate_premature_free(6)
        got = cfunc(6)
        self.assertTrue(got.sum())
        self.assertPreciseEqual(expect, got)

    def test_np_ndindex(self):
        if False:
            i = 10
            return i + 15
        func = np_ndindex
        cres = compile_isolated(func, [types.int32, types.int32])
        cfunc = cres.entry_point
        self.assertPreciseEqual(cfunc(3, 4), func(3, 4))
        self.assertPreciseEqual(cfunc(3, 0), func(3, 0))
        self.assertPreciseEqual(cfunc(0, 3), func(0, 3))
        self.assertPreciseEqual(cfunc(0, 0), func(0, 0))

    def test_np_ndindex_array(self):
        if False:
            for i in range(10):
                print('nop')
        func = np_ndindex_array
        arr = np.arange(12, dtype=np.int32) + 10
        self.check_array_unary(arr, typeof(arr), func)
        arr = arr.reshape((4, 3))
        self.check_array_unary(arr, typeof(arr), func)
        arr = arr.reshape((2, 2, 3))
        self.check_array_unary(arr, typeof(arr), func)

    def test_np_ndindex_empty(self):
        if False:
            for i in range(10):
                print('nop')
        func = np_ndindex_empty
        cres = compile_isolated(func, [])
        cfunc = cres.entry_point
        self.assertPreciseEqual(cfunc(), func())

    def test_iter_next(self):
        if False:
            i = 10
            return i + 15
        func = iter_next
        arr = np.arange(12, dtype=np.int32) + 10
        self.check_array_unary(arr, typeof(arr), func)

class TestNdIter(MemoryLeakMixin, TestCase):
    """
    Test np.nditer()
    """

    def inputs(self):
        if False:
            while True:
                i = 10
        yield np.float32(100)
        yield np.array(102, dtype=np.int16)
        yield np.arange(4).astype(np.complex64)
        yield np.arange(8)[::2]
        a = np.arange(12).reshape((3, 4))
        yield a
        yield a.copy(order='F')
        a = np.arange(24).reshape((6, 4))[::2]
        yield a

    def basic_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        yield np.arange(4).astype(np.complex64)
        yield np.arange(8)[::2]
        a = np.arange(12).reshape((3, 4))
        yield a
        yield a.copy(order='F')

    def check_result(self, got, expected):
        if False:
            i = 10
            return i + 15
        self.assertEqual(set(got), set(expected), (got, expected))

    def test_nditer1(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = np_nditer1
        cfunc = jit(nopython=True)(pyfunc)
        for a in self.inputs():
            expected = pyfunc(a)
            got = cfunc(a)
            self.check_result(got, expected)

    def test_nditer2(self):
        if False:
            print('Hello World!')
        pyfunc = np_nditer2
        cfunc = jit(nopython=True)(pyfunc)
        for (a, b) in itertools.product(self.inputs(), self.inputs()):
            expected = pyfunc(a, b)
            got = cfunc(a, b)
            self.check_result(got, expected)

    def test_nditer3(self):
        if False:
            i = 10
            return i + 15
        pyfunc = np_nditer3
        cfunc = jit(nopython=True)(pyfunc)
        inputs = self.basic_inputs
        for (a, b, c) in itertools.product(inputs(), inputs(), inputs()):
            expected = pyfunc(a, b, c)
            got = cfunc(a, b, c)
            self.check_result(got, expected)

    def test_errors(self):
        if False:
            print('Hello World!')
        pyfunc = np_nditer2
        cfunc = jit(nopython=True)(pyfunc)
        self.disable_leak_check()

        def check_incompatible(a, b):
            if False:
                while True:
                    i = 10
            with self.assertRaises(ValueError) as raises:
                cfunc(a, b)
            self.assertIn('operands could not be broadcast together', str(raises.exception))
        check_incompatible(np.arange(2), np.arange(3))
        a = np.arange(12).reshape((3, 4))
        b = np.arange(3)
        check_incompatible(a, b)
if __name__ == '__main__':
    unittest.main()