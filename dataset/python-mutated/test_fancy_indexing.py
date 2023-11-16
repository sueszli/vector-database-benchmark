import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase

def getitem_usecase(a, b):
    if False:
        i = 10
        return i + 15
    return a[b]

def setitem_usecase(a, idx, b):
    if False:
        return 10
    a[idx] = b

def np_take(A, indices):
    if False:
        i = 10
        return i + 15
    return np.take(A, indices)

def np_take_kws(A, indices, axis):
    if False:
        print('Hello World!')
    return np.take(A, indices, axis=axis)

class TestFancyIndexing(MemoryLeakMixin, TestCase):

    def generate_advanced_indices(self, N, many=True):
        if False:
            i = 10
            return i + 15
        choices = [np.int16([0, N - 1, -2])]
        if many:
            choices += [np.uint16([0, 1, N - 1]), np.bool_([0, 1, 1, 0])]
        return choices

    def generate_basic_index_tuples(self, N, maxdim, many=True):
        if False:
            return 10
        '\n        Generate basic index tuples with 0 to *maxdim* items.\n        '
        if many:
            choices = [slice(None, None, None), slice(1, N - 1, None), slice(0, None, 2), slice(N - 1, None, -2), slice(-N + 1, -1, None), slice(-1, -N, -2)]
        else:
            choices = [slice(0, N - 1, None), slice(-1, -N, -2)]
        for ndim in range(maxdim + 1):
            for tup in itertools.product(choices, repeat=ndim):
                yield tup

    def generate_advanced_index_tuples(self, N, maxdim, many=True):
        if False:
            return 10
        '\n        Generate advanced index tuples by generating basic index tuples\n        and adding a single advanced index item.\n        '
        choices = list(self.generate_advanced_indices(N, many=many))
        for i in range(maxdim + 1):
            for tup in self.generate_basic_index_tuples(N, maxdim - 1, many):
                for adv in choices:
                    yield (tup[:i] + (adv,) + tup[i:])

    def generate_advanced_index_tuples_with_ellipsis(self, N, maxdim, many=True):
        if False:
            while True:
                i = 10
        '\n        Same as generate_advanced_index_tuples(), but also insert an\n        ellipsis at various points.\n        '
        for tup in self.generate_advanced_index_tuples(N, maxdim, many):
            for i in range(len(tup) + 1):
                yield (tup[:i] + (Ellipsis,) + tup[i:])

    def check_getitem_indices(self, arr, indices):
        if False:
            return 10
        pyfunc = getitem_usecase
        cfunc = jit(nopython=True)(pyfunc)
        orig = arr.copy()
        orig_base = arr.base or arr
        for index in indices:
            expected = pyfunc(arr, index)
            assert expected.base is not orig_base
            got = cfunc(arr, index)
            self.assertEqual(got.shape, expected.shape)
            self.assertEqual(got.dtype, expected.dtype)
            np.testing.assert_equal(got, expected)
            if got.size:
                got.fill(42)
                np.testing.assert_equal(arr, orig)

    def test_getitem_tuple(self):
        if False:
            return 10
        N = 4
        ndim = 3
        arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32)
        indices = self.generate_advanced_index_tuples(N, ndim)
        self.check_getitem_indices(arr, indices)

    def test_getitem_tuple_and_ellipsis(self):
        if False:
            return 10
        N = 4
        ndim = 3
        arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32)
        indices = self.generate_advanced_index_tuples_with_ellipsis(N, ndim, many=False)
        self.check_getitem_indices(arr, indices)

    def test_ellipsis_getsetitem(self):
        if False:
            for i in range(10):
                print('nop')

        @jit(nopython=True)
        def foo(arr, v):
            if False:
                i = 10
                return i + 15
            arr[..., 0] = arr[..., 1]
        arr = np.arange(2)
        foo(arr, 1)
        self.assertEqual(arr[0], arr[1])

    def test_getitem_array(self):
        if False:
            for i in range(10):
                print('nop')
        N = 4
        ndim = 3
        arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32)
        indices = self.generate_advanced_indices(N)
        self.check_getitem_indices(arr, indices)

    def check_setitem_indices(self, arr, indices):
        if False:
            print('Hello World!')
        pyfunc = setitem_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for index in indices:
            src = arr[index]
            expected = np.zeros_like(arr)
            got = np.zeros_like(arr)
            pyfunc(expected, index, src)
            cfunc(got, index, src)
            self.assertEqual(got.shape, expected.shape)
            self.assertEqual(got.dtype, expected.dtype)
            np.testing.assert_equal(got, expected)

    def test_setitem_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        N = 4
        ndim = 3
        arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32)
        indices = self.generate_advanced_index_tuples(N, ndim)
        self.check_setitem_indices(arr, indices)

    def test_setitem_tuple_and_ellipsis(self):
        if False:
            print('Hello World!')
        N = 4
        ndim = 3
        arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32)
        indices = self.generate_advanced_index_tuples_with_ellipsis(N, ndim, many=False)
        self.check_setitem_indices(arr, indices)

    def test_setitem_array(self):
        if False:
            return 10
        N = 4
        ndim = 3
        arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32) + 10
        indices = self.generate_advanced_indices(N)
        self.check_setitem_indices(arr, indices)

    def test_setitem_0d(self):
        if False:
            print('Hello World!')
        pyfunc = setitem_usecase
        cfunc = jit(nopython=True)(pyfunc)
        inps = [(np.zeros(3), np.array(3.14)), (np.zeros(2), np.array(2)), (np.zeros(3, dtype=np.int64), np.array(3, dtype=np.int64)), (np.zeros(3, dtype=np.float64), np.array(1, dtype=np.int64)), (np.zeros(5, dtype='<U3'), np.array('abc')), (np.zeros((3,), dtype='<U3'), np.array('a')), (np.array(['abc', 'def', 'ghi'], dtype='<U3'), np.array('WXYZ', dtype='<U4')), (np.zeros(3, dtype=complex), np.array(2 + 3j, dtype=complex))]
        for (x1, v) in inps:
            x2 = x1.copy()
            pyfunc(x1, 0, v)
            cfunc(x2, 0, v)
            self.assertPreciseEqual(x1, x2)

    def test_np_take(self):
        if False:
            return 10
        pyfunc = np_take
        cfunc = jit(nopython=True)(pyfunc)

        def check(arr, ind):
            if False:
                print('Hello World!')
            expected = pyfunc(arr, ind)
            got = cfunc(arr, ind)
            self.assertPreciseEqual(expected, got)
            if hasattr(expected, 'order'):
                self.assertEqual(expected.order == got.order)
        test_indices = []
        test_indices.append(1)
        test_indices.append(np.array([1, 5, 1, 11, 3]))
        test_indices.append(np.array([[[1], [5]], [[1], [11]]]))
        test_indices.append([1, 5, 1, 11, 3])
        test_indices.append((1, 5, 1))
        test_indices.append(((1, 5, 1), (11, 3, 2)))
        for dt in [np.int64, np.complex128]:
            A = np.arange(12, dtype=dt).reshape((4, 3))
            for ind in test_indices:
                check(A, ind)
        szA = A.size
        illegal_indices = [szA, -szA - 1, np.array(szA), np.array(-szA - 1), [szA], [-szA - 1]]
        for x in illegal_indices:
            with self.assertRaises(IndexError):
                cfunc(A, x)
        with self.assertRaises(TypingError):
            cfunc(A, [1.7])
        with self.assertRaises(TypingError):
            take_kws = jit(nopython=True)(np_take_kws)
            take_kws(A, 1, 1)
        with self.assertRaises(TypingError):
            take_kws = jit(nopython=True)(np_take_kws)
            take_kws(A, 1, axis=1)
        self.disable_leak_check()

    def test_newaxis(self):
        if False:
            while True:
                i = 10

        @njit
        def np_new_axis_getitem(a, idx):
            if False:
                while True:
                    i = 10
            return a[idx]

        @njit
        def np_new_axis_setitem(a, idx, item):
            if False:
                while True:
                    i = 10
            a[idx] = item
            return a
        a = np.arange(4 * 5 * 6 * 7).reshape((4, 5, 6, 7))
        idx_cases = [(slice(None), np.newaxis), (np.newaxis, slice(None)), (slice(1), np.newaxis, np.array([1, 2, 1])), (np.newaxis, np.array([1, 2, 1]), slice(None)), (slice(1), Ellipsis, np.newaxis, np.array([1, 2, 1])), (np.array([1, 2, 1]), np.newaxis, Ellipsis), (np.newaxis, slice(1), np.newaxis, np.array([1, 2, 1])), (np.array([1, 2, 1]), Ellipsis, None, np.newaxis), (np.newaxis, slice(1), Ellipsis, np.newaxis, np.array([1, 2, 1])), (np.array([1, 2, 1]), np.newaxis, np.newaxis, Ellipsis), (np.newaxis, np.array([1, 2, 1]), np.newaxis, Ellipsis), (slice(3), np.array([1, 2, 1]), np.newaxis, None), (np.newaxis, np.array([1, 2, 1]), Ellipsis, None)]
        pyfunc_getitem = np_new_axis_getitem.py_func
        cfunc_getitem = np_new_axis_getitem
        pyfunc_setitem = np_new_axis_setitem.py_func
        cfunc_setitem = np_new_axis_setitem
        for idx in idx_cases:
            expected = pyfunc_getitem(a, idx)
            got = cfunc_getitem(a, idx)
            np.testing.assert_equal(expected, got)
            a_empty = np.zeros_like(a)
            item = a[idx]
            expected = pyfunc_setitem(a_empty.copy(), idx, item)
            got = cfunc_setitem(a_empty.copy(), idx, item)
            np.testing.assert_equal(expected, got)

class TestFancyIndexingMultiDim(MemoryLeakMixin, TestCase):
    shape = (5, 6, 7, 8, 9, 10)
    indexing_cases = [(slice(4, 5), 3, np.array([0, 1, 3, 4, 2]), 1), (3, np.array([0, 1, 3, 4, 2]), slice(None), slice(4)), (Ellipsis, 1, np.array([0, 1, 3, 4, 2])), (np.array([0, 1, 3, 4, 2]), 3, Ellipsis), (Ellipsis, 1, np.array([0, 1, 3, 4, 2]), 3, slice(1, 5)), (np.array([0, 1, 3, 4, 2]), 3, Ellipsis, slice(1, 5)), (slice(4, 5), 3, np.array([True, False, True, False, True, False, False]), 1), (3, np.array([True, False, True, False, True, False]), slice(None), slice(4))]

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.rng = np.random.default_rng(1)

    def generate_random_indices(self):
        if False:
            while True:
                i = 10
        N = min(self.shape)
        slice_choices = [slice(None, None, None), slice(1, N - 1, None), slice(0, None, 2), slice(N - 1, None, -2), slice(-N + 1, -1, None), slice(-1, -N, -2), slice(0, N - 1, None), slice(-1, -N, -2)]
        integer_choices = list(np.arange(N))
        indices = []
        K = 20
        for _ in range(K):
            array_idx = self.rng.integers(0, 5, size=15)
            curr_idx = self.rng.choice(slice_choices, size=4).tolist()
            _array_idx = self.rng.choice(4)
            curr_idx[_array_idx] = array_idx
            indices.append(tuple(curr_idx))
        for _ in range(K):
            array_idx = self.rng.integers(0, 5, size=15)
            curr_idx = self.rng.choice(integer_choices, size=4).tolist()
            _array_idx = self.rng.choice(4)
            curr_idx[_array_idx] = array_idx
            indices.append(tuple(curr_idx))
        for _ in range(K):
            array_idx = self.rng.integers(0, 5, size=15)
            curr_idx = self.rng.choice(slice_choices, size=4).tolist()
            _array_idx = self.rng.choice(4, size=2, replace=False)
            curr_idx[_array_idx[0]] = array_idx
            curr_idx[_array_idx[1]] = Ellipsis
            indices.append(tuple(curr_idx))
        for _ in range(K):
            array_idx = self.rng.integers(0, 5, size=15)
            curr_idx = self.rng.choice(slice_choices, size=4).tolist()
            _array_idx = self.rng.choice(4)
            bool_arr_shape = self.shape[_array_idx]
            curr_idx[_array_idx] = np.array(self.rng.choice(2, size=bool_arr_shape), dtype=bool)
            indices.append(tuple(curr_idx))
        return indices

    def check_getitem_indices(self, arr_shape, index):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def numba_get_item(array, idx):
            if False:
                while True:
                    i = 10
            return array[idx]
        arr = np.random.randint(0, 11, size=arr_shape)
        get_item = numba_get_item.py_func
        orig_base = arr.base or arr
        expected = get_item(arr, index)
        got = numba_get_item(arr, index)
        self.assertIsNot(expected.base, orig_base)
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, expected.dtype)
        np.testing.assert_equal(got, expected)
        self.assertFalse(np.may_share_memory(got, expected))

    def check_setitem_indices(self, arr_shape, index):
        if False:
            print('Hello World!')

        @njit
        def set_item(array, idx, item):
            if False:
                print('Hello World!')
            array[idx] = item
        arr = np.random.randint(0, 11, size=arr_shape)
        src = arr[index]
        expected = np.zeros_like(arr)
        got = np.zeros_like(arr)
        set_item.py_func(expected, index, src)
        set_item(got, index, src)
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, expected.dtype)
        np.testing.assert_equal(got, expected)

    def test_getitem(self):
        if False:
            while True:
                i = 10
        indices = self.indexing_cases.copy()
        indices += self.generate_random_indices()
        for idx in indices:
            with self.subTest(idx=idx):
                self.check_getitem_indices(self.shape, idx)

    def test_setitem(self):
        if False:
            print('Hello World!')
        indices = self.indexing_cases.copy()
        indices += self.generate_random_indices()
        for idx in indices:
            with self.subTest(idx=idx):
                self.check_setitem_indices(self.shape, idx)

    def test_unsupported_condition_exceptions(self):
        if False:
            while True:
                i = 10
        err_idx_cases = [('Multi-dimensional indices are not supported.', (0, 3, np.array([[1, 2], [2, 3]]))), ('Using more than one non-scalar array index is unsupported.', (0, 3, np.array([1, 2]), np.array([1, 2]))), ('Using more than one indexing subspace is unsupported.' + ' An indexing subspace is a group of one or more consecutive' + ' indices comprising integer or array types.', (0, np.array([1, 2]), slice(None), 3, 4))]
        for (err, idx) in err_idx_cases:
            with self.assertRaises(TypingError) as raises:
                self.check_getitem_indices(self.shape, idx)
            self.assertIn(err, str(raises.exception))
if __name__ == '__main__':
    unittest.main()