import sys
import pytest
import numpy as np
from numpy.testing import assert_, assert_raises, assert_array_equal, HAS_REFCOUNT

class TestTake:

    def test_simple(self):
        if False:
            while True:
                i = 10
        a = [[1, 2], [3, 4]]
        a_str = [[b'1', b'2'], [b'3', b'4']]
        modes = ['raise', 'wrap', 'clip']
        indices = [-1, 4]
        index_arrays = [np.empty(0, dtype=np.intp), np.empty(tuple(), dtype=np.intp), np.empty((1, 1), dtype=np.intp)]
        real_indices = {'raise': {-1: 1, 4: IndexError}, 'wrap': {-1: 1, 4: 0}, 'clip': {-1: 0, 4: 1}}
        types = (int, object, np.dtype([('', 'i2', 3)]))
        for t in types:
            ta = np.array(a if np.issubdtype(t, np.number) else a_str, dtype=t)
            tresult = list(ta.T.copy())
            for index_array in index_arrays:
                if index_array.size != 0:
                    tresult[0].shape = (2,) + index_array.shape
                    tresult[1].shape = (2,) + index_array.shape
                for mode in modes:
                    for index in indices:
                        real_index = real_indices[mode][index]
                        if real_index is IndexError and index_array.size != 0:
                            index_array.put(0, index)
                            assert_raises(IndexError, ta.take, index_array, mode=mode, axis=1)
                        elif index_array.size != 0:
                            index_array.put(0, index)
                            res = ta.take(index_array, mode=mode, axis=1)
                            assert_array_equal(res, tresult[real_index])
                        else:
                            res = ta.take(index_array, mode=mode, axis=1)
                            assert_(res.shape == (2,) + index_array.shape)

    def test_refcounting(self):
        if False:
            return 10
        objects = [object() for i in range(10)]
        for mode in ('raise', 'clip', 'wrap'):
            a = np.array(objects)
            b = np.array([2, 2, 4, 5, 3, 5])
            a.take(b, out=a[:6], mode=mode)
            del a
            if HAS_REFCOUNT:
                assert_(all((sys.getrefcount(o) == 3 for o in objects)))
            a = np.array(objects * 2)[::2]
            a.take(b, out=a[:6], mode=mode)
            del a
            if HAS_REFCOUNT:
                assert_(all((sys.getrefcount(o) == 3 for o in objects)))

    def test_unicode_mode(self):
        if False:
            for i in range(10):
                print('nop')
        d = np.arange(10)
        k = b'\xc3\xa4'.decode('UTF8')
        assert_raises(ValueError, d.take, 5, mode=k)

    def test_empty_partition(self):
        if False:
            for i in range(10):
                print('nop')
        a_original = np.array([0, 2, 4, 6, 8, 10])
        a = a_original.copy()
        a.partition(np.array([], dtype=np.int16))
        assert_array_equal(a, a_original)

    def test_empty_argpartition(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.array([0, 2, 4, 6, 8, 10])
        a = a.argpartition(np.array([], dtype=np.int16))
        b = np.array([0, 1, 2, 3, 4, 5])
        assert_array_equal(a, b)

class TestPutMask:

    @pytest.mark.parametrize('dtype', list(np.typecodes['All']) + ['i,O'])
    def test_simple(self, dtype):
        if False:
            i = 10
            return i + 15
        if dtype.lower() == 'm':
            dtype += '8[ns]'
        vals = np.arange(1001).astype(dtype=dtype)
        mask = np.random.randint(2, size=1000).astype(bool)
        arr = np.zeros(1000, dtype=vals.dtype)
        zeros = arr.copy()
        np.putmask(arr, mask, vals)
        assert_array_equal(arr[mask], vals[:len(mask)][mask])
        assert_array_equal(arr[~mask], zeros[~mask])

    @pytest.mark.parametrize('dtype', list(np.typecodes['All'])[1:] + ['i,O'])
    @pytest.mark.parametrize('mode', ['raise', 'wrap', 'clip'])
    def test_empty(self, dtype, mode):
        if False:
            for i in range(10):
                print('nop')
        arr = np.zeros(1000, dtype=dtype)
        arr_copy = arr.copy()
        mask = np.random.randint(2, size=1000).astype(bool)
        np.put(arr, mask, [])
        assert_array_equal(arr, arr_copy)

class TestPut:

    @pytest.mark.parametrize('dtype', list(np.typecodes['All'])[1:] + ['i,O'])
    @pytest.mark.parametrize('mode', ['raise', 'wrap', 'clip'])
    def test_simple(self, dtype, mode):
        if False:
            i = 10
            return i + 15
        if dtype.lower() == 'm':
            dtype += '8[ns]'
        vals = np.arange(1001).astype(dtype=dtype)
        arr = np.zeros(1000, dtype=vals.dtype)
        zeros = arr.copy()
        if mode == 'clip':
            indx = np.random.permutation(len(arr) - 2)[:-500] + 1
            indx[-1] = 0
            indx[-2] = len(arr) - 1
            indx_put = indx.copy()
            indx_put[-1] = -1389
            indx_put[-2] = 1321
        else:
            indx = np.random.permutation(len(arr) - 3)[:-500]
            indx_put = indx
            if mode == 'wrap':
                indx_put = indx_put + len(arr)
        np.put(arr, indx_put, vals, mode=mode)
        assert_array_equal(arr[indx], vals[:len(indx)])
        untouched = np.ones(len(arr), dtype=bool)
        untouched[indx] = False
        assert_array_equal(arr[untouched], zeros[:untouched.sum()])

    @pytest.mark.parametrize('dtype', list(np.typecodes['All'])[1:] + ['i,O'])
    @pytest.mark.parametrize('mode', ['raise', 'wrap', 'clip'])
    def test_empty(self, dtype, mode):
        if False:
            for i in range(10):
                print('nop')
        arr = np.zeros(1000, dtype=dtype)
        arr_copy = arr.copy()
        np.put(arr, [1, 2, 3], [])
        assert_array_equal(arr, arr_copy)