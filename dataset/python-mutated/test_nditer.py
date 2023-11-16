import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy._core.umath as ncu
import numpy._core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import assert_, assert_equal, assert_array_equal, assert_raises, IS_WASM, HAS_REFCOUNT, suppress_warnings, break_cycles

def iter_multi_index(i):
    if False:
        for i in range(10):
            print('nop')
    ret = []
    while not i.finished:
        ret.append(i.multi_index)
        i.iternext()
    return ret

def iter_indices(i):
    if False:
        for i in range(10):
            print('nop')
    ret = []
    while not i.finished:
        ret.append(i.index)
        i.iternext()
    return ret

def iter_iterindices(i):
    if False:
        while True:
            i = 10
    ret = []
    while not i.finished:
        ret.append(i.iterindex)
        i.iternext()
    return ret

@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_iter_refcount():
    if False:
        for i in range(10):
            print('nop')
    a = arange(6)
    dt = np.dtype('f4').newbyteorder()
    rc_a = sys.getrefcount(a)
    rc_dt = sys.getrefcount(dt)
    with nditer(a, [], [['readwrite', 'updateifcopy']], casting='unsafe', op_dtypes=[dt]) as it:
        assert_(not it.iterationneedsapi)
        assert_(sys.getrefcount(a) > rc_a)
        assert_(sys.getrefcount(dt) > rc_dt)
    it = None
    assert_equal(sys.getrefcount(a), rc_a)
    assert_equal(sys.getrefcount(dt), rc_dt)
    a = arange(6, dtype='f4')
    dt = np.dtype('f4')
    rc_a = sys.getrefcount(a)
    rc_dt = sys.getrefcount(dt)
    it = nditer(a, [], [['readwrite']], op_dtypes=[dt])
    rc2_a = sys.getrefcount(a)
    rc2_dt = sys.getrefcount(dt)
    it2 = it.copy()
    assert_(sys.getrefcount(a) > rc2_a)
    assert_(sys.getrefcount(dt) > rc2_dt)
    it = None
    assert_equal(sys.getrefcount(a), rc2_a)
    assert_equal(sys.getrefcount(dt), rc2_dt)
    it2 = None
    assert_equal(sys.getrefcount(a), rc_a)
    assert_equal(sys.getrefcount(dt), rc_dt)
    del it2

def test_iter_best_order():
    if False:
        for i in range(10):
            print('nop')
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        a = arange(np.prod(shape))
        for dirs in range(2 ** len(shape)):
            dirs_index = [slice(None)] * len(shape)
            for bit in range(len(shape)):
                if 2 ** bit & dirs:
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)
            aview = a.reshape(shape)[dirs_index]
            i = nditer(aview, [], [['readonly']])
            assert_equal([x for x in i], a)
            i = nditer(aview.T, [], [['readonly']])
            assert_equal([x for x in i], a)
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), [], [['readonly']])
                assert_equal([x for x in i], a)

def test_iter_c_order():
    if False:
        while True:
            i = 10
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        a = arange(np.prod(shape))
        for dirs in range(2 ** len(shape)):
            dirs_index = [slice(None)] * len(shape)
            for bit in range(len(shape)):
                if 2 ** bit & dirs:
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)
            aview = a.reshape(shape)[dirs_index]
            i = nditer(aview, order='C')
            assert_equal([x for x in i], aview.ravel(order='C'))
            i = nditer(aview.T, order='C')
            assert_equal([x for x in i], aview.T.ravel(order='C'))
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), order='C')
                assert_equal([x for x in i], aview.swapaxes(0, 1).ravel(order='C'))

def test_iter_f_order():
    if False:
        while True:
            i = 10
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        a = arange(np.prod(shape))
        for dirs in range(2 ** len(shape)):
            dirs_index = [slice(None)] * len(shape)
            for bit in range(len(shape)):
                if 2 ** bit & dirs:
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)
            aview = a.reshape(shape)[dirs_index]
            i = nditer(aview, order='F')
            assert_equal([x for x in i], aview.ravel(order='F'))
            i = nditer(aview.T, order='F')
            assert_equal([x for x in i], aview.T.ravel(order='F'))
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), order='F')
                assert_equal([x for x in i], aview.swapaxes(0, 1).ravel(order='F'))

def test_iter_c_or_f_order():
    if False:
        print('Hello World!')
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        a = arange(np.prod(shape))
        for dirs in range(2 ** len(shape)):
            dirs_index = [slice(None)] * len(shape)
            for bit in range(len(shape)):
                if 2 ** bit & dirs:
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)
            aview = a.reshape(shape)[dirs_index]
            i = nditer(aview, order='A')
            assert_equal([x for x in i], aview.ravel(order='A'))
            i = nditer(aview.T, order='A')
            assert_equal([x for x in i], aview.T.ravel(order='A'))
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), order='A')
                assert_equal([x for x in i], aview.swapaxes(0, 1).ravel(order='A'))

def test_nditer_multi_index_set():
    if False:
        i = 10
        return i + 15
    a = np.arange(6).reshape(2, 3)
    it = np.nditer(a, flags=['multi_index'])
    it.multi_index = (0, 2)
    assert_equal([i for i in it], [2, 3, 4, 5])

@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_nditer_multi_index_set_refcount():
    if False:
        print('Hello World!')
    index = 0
    i = np.nditer(np.array([111, 222, 333, 444]), flags=['multi_index'])
    start_count = sys.getrefcount(index)
    i.multi_index = (index,)
    end_count = sys.getrefcount(index)
    assert_equal(start_count, end_count)

def test_iter_best_order_multi_index_1d():
    if False:
        return 10
    a = arange(4)
    i = nditer(a, ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0,), (1,), (2,), (3,)])
    i = nditer(a[::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(3,), (2,), (1,), (0,)])

def test_iter_best_order_multi_index_2d():
    if False:
        print('Hello World!')
    a = arange(6)
    i = nditer(a.reshape(2, 3), ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
    i = nditer(a.reshape(2, 3).copy(order='F'), ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)])
    i = nditer(a.reshape(2, 3)[::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(1, 0), (1, 1), (1, 2), (0, 0), (0, 1), (0, 2)])
    i = nditer(a.reshape(2, 3)[:, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 2), (0, 1), (0, 0), (1, 2), (1, 1), (1, 0)])
    i = nditer(a.reshape(2, 3)[::-1, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2)])
    i = nditer(a.reshape(2, 3).copy(order='F')[:, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 2), (1, 2), (0, 1), (1, 1), (0, 0), (1, 0)])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(1, 2), (0, 2), (1, 1), (0, 1), (1, 0), (0, 0)])

def test_iter_best_order_multi_index_3d():
    if False:
        return 10
    a = arange(12)
    i = nditer(a.reshape(2, 3, 2), ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1)])
    i = nditer(a.reshape(2, 3, 2).copy(order='F'), ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), (1, 2, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1), (1, 2, 1)])
    i = nditer(a.reshape(2, 3, 2)[::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1)])
    i = nditer(a.reshape(2, 3, 2)[:, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 2, 0), (0, 2, 1), (0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 2, 0), (1, 2, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)])
    i = nditer(a.reshape(2, 3, 2)[:, :, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 0, 1), (0, 0, 0), (0, 1, 1), (0, 1, 0), (0, 2, 1), (0, 2, 0), (1, 0, 1), (1, 0, 0), (1, 1, 1), (1, 1, 0), (1, 2, 1), (1, 2, 0)])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 2, 0), (0, 2, 0), (1, 0, 1), (0, 0, 1), (1, 1, 1), (0, 1, 1), (1, 2, 1), (0, 2, 1)])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 2, 0), (1, 2, 0), (0, 1, 0), (1, 1, 0), (0, 0, 0), (1, 0, 0), (0, 2, 1), (1, 2, 1), (0, 1, 1), (1, 1, 1), (0, 0, 1), (1, 0, 1)])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, :, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1), (1, 2, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), (1, 2, 0)])

def test_iter_best_order_c_index_1d():
    if False:
        for i in range(10):
            print('nop')
    a = arange(4)
    i = nditer(a, ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3])
    i = nditer(a[::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [3, 2, 1, 0])

def test_iter_best_order_c_index_2d():
    if False:
        print('Hello World!')
    a = arange(6)
    i = nditer(a.reshape(2, 3), ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3, 4, 5])
    i = nditer(a.reshape(2, 3).copy(order='F'), ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 3, 1, 4, 2, 5])
    i = nditer(a.reshape(2, 3)[::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [3, 4, 5, 0, 1, 2])
    i = nditer(a.reshape(2, 3)[:, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [2, 1, 0, 5, 4, 3])
    i = nditer(a.reshape(2, 3)[::-1, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [5, 4, 3, 2, 1, 0])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [3, 0, 4, 1, 5, 2])
    i = nditer(a.reshape(2, 3).copy(order='F')[:, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [2, 5, 1, 4, 0, 3])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [5, 2, 4, 1, 3, 0])

def test_iter_best_order_c_index_3d():
    if False:
        return 10
    a = arange(12)
    i = nditer(a.reshape(2, 3, 2), ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    i = nditer(a.reshape(2, 3, 2).copy(order='F'), ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11])
    i = nditer(a.reshape(2, 3, 2)[::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])
    i = nditer(a.reshape(2, 3, 2)[:, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7])
    i = nditer(a.reshape(2, 3, 2)[:, :, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [6, 0, 8, 2, 10, 4, 7, 1, 9, 3, 11, 5])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [4, 10, 2, 8, 0, 6, 5, 11, 3, 9, 1, 7])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, :, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [1, 7, 3, 9, 5, 11, 0, 6, 2, 8, 4, 10])

def test_iter_best_order_f_index_1d():
    if False:
        while True:
            i = 10
    a = arange(4)
    i = nditer(a, ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3])
    i = nditer(a[::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [3, 2, 1, 0])

def test_iter_best_order_f_index_2d():
    if False:
        return 10
    a = arange(6)
    i = nditer(a.reshape(2, 3), ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 2, 4, 1, 3, 5])
    i = nditer(a.reshape(2, 3).copy(order='F'), ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3, 4, 5])
    i = nditer(a.reshape(2, 3)[::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [1, 3, 5, 0, 2, 4])
    i = nditer(a.reshape(2, 3)[:, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [4, 2, 0, 5, 3, 1])
    i = nditer(a.reshape(2, 3)[::-1, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [5, 3, 1, 4, 2, 0])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [1, 0, 3, 2, 5, 4])
    i = nditer(a.reshape(2, 3).copy(order='F')[:, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [4, 5, 2, 3, 0, 1])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [5, 4, 3, 2, 1, 0])

def test_iter_best_order_f_index_3d():
    if False:
        while True:
            i = 10
    a = arange(12)
    i = nditer(a.reshape(2, 3, 2), ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11])
    i = nditer(a.reshape(2, 3, 2).copy(order='F'), ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    i = nditer(a.reshape(2, 3, 2)[::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [1, 7, 3, 9, 5, 11, 0, 6, 2, 8, 4, 10])
    i = nditer(a.reshape(2, 3, 2)[:, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [4, 10, 2, 8, 0, 6, 5, 11, 3, 9, 1, 7])
    i = nditer(a.reshape(2, 3, 2)[:, :, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [6, 0, 8, 2, 10, 4, 7, 1, 9, 3, 11, 5])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, :, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])

def test_iter_no_inner_full_coalesce():
    if False:
        i = 10
        return i + 15
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        size = np.prod(shape)
        a = arange(size)
        for dirs in range(2 ** len(shape)):
            dirs_index = [slice(None)] * len(shape)
            for bit in range(len(shape)):
                if 2 ** bit & dirs:
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)
            aview = a.reshape(shape)[dirs_index]
            i = nditer(aview, ['external_loop'], [['readonly']])
            assert_equal(i.ndim, 1)
            assert_equal(i[0].shape, (size,))
            i = nditer(aview.T, ['external_loop'], [['readonly']])
            assert_equal(i.ndim, 1)
            assert_equal(i[0].shape, (size,))
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), ['external_loop'], [['readonly']])
                assert_equal(i.ndim, 1)
                assert_equal(i[0].shape, (size,))

def test_iter_no_inner_dim_coalescing():
    if False:
        return 10
    a = arange(24).reshape(2, 3, 4)[:, :, :-1]
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 2)
    assert_equal(i[0].shape, (3,))
    a = arange(24).reshape(2, 3, 4)[:, :-1, :]
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 2)
    assert_equal(i[0].shape, (8,))
    a = arange(24).reshape(2, 3, 4)[:-1, :, :]
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 1)
    assert_equal(i[0].shape, (12,))
    a = arange(24).reshape(1, 1, 2, 1, 1, 3, 1, 1, 4, 1, 1)
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 1)
    assert_equal(i[0].shape, (24,))

def test_iter_dim_coalescing():
    if False:
        return 10
    a = arange(24).reshape(2, 3, 4)
    i = nditer(a, ['multi_index'], [['readonly']])
    assert_equal(i.ndim, 3)
    a3d = arange(24).reshape(2, 3, 4)
    i = nditer(a3d, ['c_index'], [['readonly']])
    assert_equal(i.ndim, 1)
    i = nditer(a3d.swapaxes(0, 1), ['c_index'], [['readonly']])
    assert_equal(i.ndim, 3)
    i = nditer(a3d.T, ['c_index'], [['readonly']])
    assert_equal(i.ndim, 3)
    i = nditer(a3d.T, ['f_index'], [['readonly']])
    assert_equal(i.ndim, 1)
    i = nditer(a3d.T.swapaxes(0, 1), ['f_index'], [['readonly']])
    assert_equal(i.ndim, 3)
    a3d = arange(24).reshape(2, 3, 4)
    i = nditer(a3d, order='C')
    assert_equal(i.ndim, 1)
    i = nditer(a3d.T, order='C')
    assert_equal(i.ndim, 3)
    i = nditer(a3d, order='F')
    assert_equal(i.ndim, 3)
    i = nditer(a3d.T, order='F')
    assert_equal(i.ndim, 1)
    i = nditer(a3d, order='A')
    assert_equal(i.ndim, 1)
    i = nditer(a3d.T, order='A')
    assert_equal(i.ndim, 1)

def test_iter_broadcasting():
    if False:
        return 10
    i = nditer([arange(6), np.int32(2)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (6,))
    i = nditer([arange(6).reshape(2, 3), np.int32(2)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (2, 3))
    i = nditer([arange(6).reshape(2, 3), arange(3)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (2, 3))
    i = nditer([arange(2).reshape(2, 1), arange(3)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (2, 3))
    i = nditer([arange(2).reshape(2, 1), arange(3).reshape(1, 3)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (2, 3))
    i = nditer([np.int32(2), arange(24).reshape(4, 2, 3)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(3), arange(24).reshape(4, 2, 3)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(3), arange(8).reshape(4, 2, 1)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(6).reshape(2, 3), arange(24).reshape(4, 2, 3)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(2).reshape(2, 1), arange(24).reshape(4, 2, 3)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(3).reshape(1, 3), arange(8).reshape(4, 2, 1)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(2).reshape(1, 2, 1), arange(3).reshape(1, 1, 3), arange(4).reshape(4, 1, 1)], ['multi_index'], [['readonly']] * 3)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(6).reshape(1, 2, 3), arange(4).reshape(4, 1, 1)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(24).reshape(4, 2, 3), arange(12).reshape(4, 1, 3)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))

def test_iter_itershape():
    if False:
        while True:
            i = 10
    a = np.arange(6, dtype='i2').reshape(2, 3)
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']], op_axes=[[0, 1, None], None], itershape=(-1, -1, 4))
    assert_equal(i.operands[1].shape, (2, 3, 4))
    assert_equal(i.operands[1].strides, (24, 8, 2))
    i = nditer([a.T, None], [], [['readonly'], ['writeonly', 'allocate']], op_axes=[[0, 1, None], None], itershape=(-1, -1, 4))
    assert_equal(i.operands[1].shape, (3, 2, 4))
    assert_equal(i.operands[1].strides, (8, 24, 2))
    i = nditer([a.T, None], [], [['readonly'], ['writeonly', 'allocate']], order='F', op_axes=[[0, 1, None], None], itershape=(-1, -1, 4))
    assert_equal(i.operands[1].shape, (3, 2, 4))
    assert_equal(i.operands[1].strides, (2, 6, 12))
    assert_raises(ValueError, nditer, [a, None], [], [['readonly'], ['writeonly', 'allocate']], op_axes=[[0, 1, None], None], itershape=(-1, 1, 4))
    i = np.nditer([np.ones(2), None, None], itershape=(2,))

def test_iter_broadcasting_errors():
    if False:
        return 10
    assert_raises(ValueError, nditer, [arange(2), arange(3)], [], [['readonly']] * 2)
    assert_raises(ValueError, nditer, [arange(6).reshape(2, 3), arange(2)], [], [['readonly']] * 2)
    assert_raises(ValueError, nditer, [arange(6).reshape(2, 3), arange(9).reshape(3, 3)], [], [['readonly']] * 2)
    assert_raises(ValueError, nditer, [arange(6).reshape(2, 3), arange(4).reshape(2, 2)], [], [['readonly']] * 2)
    assert_raises(ValueError, nditer, [arange(36).reshape(3, 3, 4), arange(24).reshape(2, 3, 4)], [], [['readonly']] * 2)
    assert_raises(ValueError, nditer, [arange(8).reshape(2, 4, 1), arange(24).reshape(2, 3, 4)], [], [['readonly']] * 2)
    try:
        nditer([arange(2).reshape(1, 2, 1), arange(3).reshape(1, 3), arange(6).reshape(2, 3)], [], [['readonly'], ['readonly'], ['writeonly', 'no_broadcast']])
        raise AssertionError('Should have raised a broadcast error')
    except ValueError as e:
        msg = str(e)
        assert_(msg.find('(2,3)') >= 0, 'Message "%s" doesn\'t contain operand shape (2,3)' % msg)
        assert_(msg.find('(1,2,3)') >= 0, 'Message "%s" doesn\'t contain broadcast shape (1,2,3)' % msg)
    try:
        nditer([arange(6).reshape(2, 3), arange(2)], [], [['readonly'], ['readonly']], op_axes=[[0, 1], [0, np.newaxis]], itershape=(4, 3))
        raise AssertionError('Should have raised a broadcast error')
    except ValueError as e:
        msg = str(e)
        assert_(msg.find('(2,3)->(2,3)') >= 0, 'Message "%s" doesn\'t contain operand shape (2,3)->(2,3)' % msg)
        assert_(msg.find('(2,)->(2,newaxis)') >= 0, ('Message "%s" doesn\'t contain remapped operand shape' + '(2,)->(2,newaxis)') % msg)
        assert_(msg.find('(4,3)') >= 0, 'Message "%s" doesn\'t contain itershape parameter (4,3)' % msg)
    try:
        nditer([np.zeros((2, 1, 1)), np.zeros((2,))], [], [['writeonly', 'no_broadcast'], ['readonly']])
        raise AssertionError('Should have raised a broadcast error')
    except ValueError as e:
        msg = str(e)
        assert_(msg.find('(2,1,1)') >= 0, 'Message "%s" doesn\'t contain operand shape (2,1,1)' % msg)
        assert_(msg.find('(2,1,2)') >= 0, 'Message "%s" doesn\'t contain the broadcast shape (2,1,2)' % msg)

def test_iter_flags_errors():
    if False:
        i = 10
        return i + 15
    a = arange(6)
    assert_raises(ValueError, nditer, [], [], [])
    assert_raises(ValueError, nditer, [a] * 100, [], [['readonly']] * 100)
    assert_raises(ValueError, nditer, [a], ['bad flag'], [['readonly']])
    assert_raises(ValueError, nditer, [a], [], [['readonly', 'bad flag']])
    assert_raises(ValueError, nditer, [a], [], [['readonly']], order='G')
    assert_raises(ValueError, nditer, [a], [], [['readonly']], casting='noon')
    assert_raises(ValueError, nditer, [a] * 3, [], [['readonly']] * 2)
    assert_raises(ValueError, nditer, a, ['c_index', 'f_index'], [['readonly']])
    assert_raises(ValueError, nditer, a, ['external_loop', 'multi_index'], [['readonly']])
    assert_raises(ValueError, nditer, a, ['external_loop', 'c_index'], [['readonly']])
    assert_raises(ValueError, nditer, a, ['external_loop', 'f_index'], [['readonly']])
    assert_raises(ValueError, nditer, a, [], [[]])
    assert_raises(ValueError, nditer, a, [], [['readonly', 'writeonly']])
    assert_raises(ValueError, nditer, a, [], [['readonly', 'readwrite']])
    assert_raises(ValueError, nditer, a, [], [['writeonly', 'readwrite']])
    assert_raises(ValueError, nditer, a, [], [['readonly', 'writeonly', 'readwrite']])
    assert_raises(TypeError, nditer, 1.5, [], [['writeonly']])
    assert_raises(TypeError, nditer, 1.5, [], [['readwrite']])
    assert_raises(TypeError, nditer, np.int32(1), [], [['writeonly']])
    assert_raises(TypeError, nditer, np.int32(1), [], [['readwrite']])
    a.flags.writeable = False
    assert_raises(ValueError, nditer, a, [], [['writeonly']])
    assert_raises(ValueError, nditer, a, [], [['readwrite']])
    a.flags.writeable = True
    i = nditer(arange(6), [], [['readonly']])
    assert_raises(ValueError, lambda i: i.multi_index, i)
    assert_raises(ValueError, lambda i: i.index, i)

    def assign_multi_index(i):
        if False:
            while True:
                i = 10
        i.multi_index = (0,)

    def assign_index(i):
        if False:
            return 10
        i.index = 0

    def assign_iterindex(i):
        if False:
            return 10
        i.iterindex = 0

    def assign_iterrange(i):
        if False:
            return 10
        i.iterrange = (0, 1)
    i = nditer(arange(6), ['external_loop'])
    assert_raises(ValueError, assign_multi_index, i)
    assert_raises(ValueError, assign_index, i)
    assert_raises(ValueError, assign_iterindex, i)
    assert_raises(ValueError, assign_iterrange, i)
    i = nditer(arange(6), ['buffered'])
    assert_raises(ValueError, assign_multi_index, i)
    assert_raises(ValueError, assign_index, i)
    assert_raises(ValueError, assign_iterrange, i)
    assert_raises(ValueError, nditer, np.array([]))

def test_iter_slice():
    if False:
        while True:
            i = 10
    (a, b, c) = (np.arange(3), np.arange(3), np.arange(3.0))
    i = nditer([a, b, c], [], ['readwrite'])
    with i:
        i[0:2] = (3, 3)
        assert_equal(a, [3, 1, 2])
        assert_equal(b, [3, 1, 2])
        assert_equal(c, [0, 1, 2])
        i[1] = 12
        assert_equal(i[0:2], [3, 12])

def test_iter_assign_mapping():
    if False:
        for i in range(10):
            print('nop')
    a = np.arange(24, dtype='f8').reshape(2, 3, 4).T
    it = np.nditer(a, [], [['readwrite', 'updateifcopy']], casting='same_kind', op_dtypes=[np.dtype('f4')])
    with it:
        it.operands[0][...] = 3
        it.operands[0][...] = 14
    assert_equal(a, 14)
    it = np.nditer(a, [], [['readwrite', 'updateifcopy']], casting='same_kind', op_dtypes=[np.dtype('f4')])
    with it:
        x = it.operands[0][-1:1]
        x[...] = 14
        it.operands[0][...] = -1234
    assert_equal(a, -1234)
    x = None
    it = None

def test_iter_nbo_align_contig():
    if False:
        for i in range(10):
            print('nop')
    a = np.arange(6, dtype='f4')
    au = a.byteswap()
    au = au.view(au.dtype.newbyteorder())
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    i = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    with i:
        assert_equal(i.dtypes[0].byteorder, a.dtype.byteorder)
        assert_equal(i.operands[0].dtype.byteorder, a.dtype.byteorder)
        assert_equal(i.operands[0], a)
        i.operands[0][:] = 2
    assert_equal(au, [2] * 6)
    del i
    a = np.arange(6, dtype='f4')
    au = a.byteswap()
    au = au.view(au.dtype.newbyteorder())
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    with nditer(au, [], [['readwrite', 'updateifcopy', 'nbo']], casting='equiv') as i:
        assert_equal(i.dtypes[0].byteorder, a.dtype.byteorder)
        assert_equal(i.operands[0].dtype.byteorder, a.dtype.byteorder)
        assert_equal(i.operands[0], a)
        i.operands[0][:] = 12345
        i.operands[0][:] = 2
    assert_equal(au, [2] * 6)
    a = np.zeros((6 * 4 + 1,), dtype='i1')[1:]
    a.dtype = 'f4'
    a[:] = np.arange(6, dtype='f4')
    assert_(not a.flags.aligned)
    i = nditer(a, [], [['readonly']])
    assert_(not i.operands[0].flags.aligned)
    assert_equal(i.operands[0], a)
    with nditer(a, [], [['readwrite', 'updateifcopy', 'aligned']]) as i:
        assert_(i.operands[0].flags.aligned)
        assert_equal(i.operands[0], a)
        i.operands[0][:] = 3
    assert_equal(a, [3] * 6)
    a = arange(12)
    i = nditer(a[:6], [], [['readonly']])
    assert_(i.operands[0].flags.contiguous)
    assert_equal(i.operands[0], a[:6])
    i = nditer(a[::2], ['buffered', 'external_loop'], [['readonly', 'contig']], buffersize=10)
    assert_(i[0].flags.contiguous)
    assert_equal(i[0], a[::2])

def test_iter_array_cast():
    if False:
        i = 10
        return i + 15
    a = np.arange(6, dtype='f4').reshape(2, 3)
    i = nditer(a, [], [['readwrite']], op_dtypes=[np.dtype('f4')])
    with i:
        assert_equal(i.operands[0], a)
        assert_equal(i.operands[0].dtype, np.dtype('f4'))
    a = np.arange(6, dtype='<f4').reshape(2, 3)
    with nditer(a, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('>f4')]) as i:
        assert_equal(i.operands[0], a)
        assert_equal(i.operands[0].dtype, np.dtype('>f4'))
    a = np.arange(24, dtype='f4').reshape(2, 3, 4).swapaxes(1, 2)
    i = nditer(a, [], [['readonly', 'copy']], casting='safe', op_dtypes=[np.dtype('f8')])
    assert_equal(i.operands[0], a)
    assert_equal(i.operands[0].dtype, np.dtype('f8'))
    assert_equal(i.operands[0].strides, (96, 8, 32))
    a = a[::-1, :, ::-1]
    i = nditer(a, [], [['readonly', 'copy']], casting='safe', op_dtypes=[np.dtype('f8')])
    assert_equal(i.operands[0], a)
    assert_equal(i.operands[0].dtype, np.dtype('f8'))
    assert_equal(i.operands[0].strides, (96, 8, 32))
    a = np.arange(24, dtype='f8').reshape(2, 3, 4).T
    with nditer(a, [], [['readwrite', 'updateifcopy']], casting='same_kind', op_dtypes=[np.dtype('f4')]) as i:
        assert_equal(i.operands[0], a)
        assert_equal(i.operands[0].dtype, np.dtype('f4'))
        assert_equal(i.operands[0].strides, (4, 16, 48))
        i.operands[0][2, 1, 1] = -12.5
        assert_(a[2, 1, 1] != -12.5)
    assert_equal(a[2, 1, 1], -12.5)
    a = np.arange(6, dtype='i4')[::-2]
    with nditer(a, [], [['writeonly', 'updateifcopy']], casting='unsafe', op_dtypes=[np.dtype('f4')]) as i:
        assert_equal(i.operands[0].dtype, np.dtype('f4'))
        assert_equal(i.operands[0].strides, (4,))
        i.operands[0][:] = [1, 2, 3]
    assert_equal(a, [1, 2, 3])

def test_iter_array_cast_errors():
    if False:
        while True:
            i = 10
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [], [['readonly']], op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [], [['readonly', 'copy']], casting='no', op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [], [['readonly', 'copy']], casting='equiv', op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, nditer, arange(2, dtype='f8'), [], [['writeonly', 'updateifcopy']], casting='no', op_dtypes=[np.dtype('f4')])
    assert_raises(TypeError, nditer, arange(2, dtype='f8'), [], [['writeonly', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    assert_raises(TypeError, nditer, arange(2, dtype='<f4'), [], [['readonly', 'copy']], casting='no', op_dtypes=[np.dtype('>f4')])
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [], [['readwrite', 'updateifcopy']], casting='safe', op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, nditer, arange(2, dtype='f8'), [], [['readwrite', 'updateifcopy']], casting='safe', op_dtypes=[np.dtype('f4')])
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [], [['readonly', 'copy']], casting='same_kind', op_dtypes=[np.dtype('i4')])
    assert_raises(TypeError, nditer, arange(2, dtype='i4'), [], [['writeonly', 'updateifcopy']], casting='same_kind', op_dtypes=[np.dtype('f4')])

def test_iter_scalar_cast():
    if False:
        while True:
            i = 10
    i = nditer(np.float32(2.5), [], [['readonly']], op_dtypes=[np.dtype('f4')])
    assert_equal(i.dtypes[0], np.dtype('f4'))
    assert_equal(i.value.dtype, np.dtype('f4'))
    assert_equal(i.value, 2.5)
    i = nditer(np.float32(2.5), [], [['readonly', 'copy']], casting='safe', op_dtypes=[np.dtype('f8')])
    assert_equal(i.dtypes[0], np.dtype('f8'))
    assert_equal(i.value.dtype, np.dtype('f8'))
    assert_equal(i.value, 2.5)
    i = nditer(np.float64(2.5), [], [['readonly', 'copy']], casting='same_kind', op_dtypes=[np.dtype('f4')])
    assert_equal(i.dtypes[0], np.dtype('f4'))
    assert_equal(i.value.dtype, np.dtype('f4'))
    assert_equal(i.value, 2.5)
    i = nditer(np.float64(3.0), [], [['readonly', 'copy']], casting='unsafe', op_dtypes=[np.dtype('i4')])
    assert_equal(i.dtypes[0], np.dtype('i4'))
    assert_equal(i.value.dtype, np.dtype('i4'))
    assert_equal(i.value, 3)
    i = nditer(3, [], [['readonly']], op_dtypes=[np.dtype('f8')])
    assert_equal(i[0].dtype, np.dtype('f8'))
    assert_equal(i[0], 3.0)

def test_iter_scalar_cast_errors():
    if False:
        for i in range(10):
            print('nop')
    assert_raises(TypeError, nditer, np.float32(2), [], [['readwrite']], op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, nditer, 2.5, [], [['readwrite']], op_dtypes=[np.dtype('f4')])
    assert_raises(TypeError, nditer, np.float64(1e+60), [], [['readonly']], casting='safe', op_dtypes=[np.dtype('f4')])
    assert_raises(TypeError, nditer, np.float32(2), [], [['readonly']], casting='same_kind', op_dtypes=[np.dtype('i4')])

def test_iter_object_arrays_basic():
    if False:
        while True:
            i = 10
    obj = {'a': 3, 'b': 'd'}
    a = np.array([[1, 2, 3], None, obj, None], dtype='O')
    if HAS_REFCOUNT:
        rc = sys.getrefcount(obj)
    assert_raises(TypeError, nditer, a)
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(obj), rc)
    i = nditer(a, ['refs_ok'], ['readonly'])
    vals = [x_[()] for x_ in i]
    assert_equal(np.array(vals, dtype='O'), a)
    (vals, i, x) = [None] * 3
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(obj), rc)
    i = nditer(a.reshape(2, 2).T, ['refs_ok', 'buffered'], ['readonly'], order='C')
    assert_(i.iterationneedsapi)
    vals = [x_[()] for x_ in i]
    assert_equal(np.array(vals, dtype='O'), a.reshape(2, 2).ravel(order='F'))
    (vals, i, x) = [None] * 3
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(obj), rc)
    i = nditer(a.reshape(2, 2).T, ['refs_ok', 'buffered'], ['readwrite'], order='C')
    with i:
        for x in i:
            x[...] = None
        (vals, i, x) = [None] * 3
    if HAS_REFCOUNT:
        assert_(sys.getrefcount(obj) == rc - 1)
    assert_equal(a, np.array([None] * 4, dtype='O'))

def test_iter_object_arrays_conversions():
    if False:
        return 10
    a = np.arange(6, dtype='O')
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'], casting='unsafe', op_dtypes='i4')
    with i:
        for x in i:
            x[...] += 1
    assert_equal(a, np.arange(6) + 1)
    a = np.arange(6, dtype='i4')
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'], casting='unsafe', op_dtypes='O')
    with i:
        for x in i:
            x[...] += 1
    assert_equal(a, np.arange(6) + 1)
    a = np.zeros((6,), dtype=[('p', 'i1'), ('a', 'O')])
    a = a['a']
    a[:] = np.arange(6)
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'], casting='unsafe', op_dtypes='i4')
    with i:
        for x in i:
            x[...] += 1
    assert_equal(a, np.arange(6) + 1)
    a = np.zeros((6,), dtype=[('p', 'i1'), ('a', 'i4')])
    a = a['a']
    a[:] = np.arange(6) + 98172488
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'], casting='unsafe', op_dtypes='O')
    with i:
        ob = i[0][()]
        if HAS_REFCOUNT:
            rc = sys.getrefcount(ob)
        for x in i:
            x[...] += 1
    if HAS_REFCOUNT:
        assert_(sys.getrefcount(ob) == rc - 1)
    assert_equal(a, np.arange(6) + 98172489)

def test_iter_common_dtype():
    if False:
        return 10
    i = nditer([array([3], dtype='f4'), array([0], dtype='f8')], ['common_dtype'], [['readonly', 'copy']] * 2, casting='safe')
    assert_equal(i.dtypes[0], np.dtype('f8'))
    assert_equal(i.dtypes[1], np.dtype('f8'))
    i = nditer([array([3], dtype='i4'), array([0], dtype='f4')], ['common_dtype'], [['readonly', 'copy']] * 2, casting='safe')
    assert_equal(i.dtypes[0], np.dtype('f8'))
    assert_equal(i.dtypes[1], np.dtype('f8'))
    i = nditer([array([3], dtype='f4'), array(0, dtype='f8')], ['common_dtype'], [['readonly', 'copy']] * 2, casting='same_kind')
    assert_equal(i.dtypes[0], np.dtype('f8'))
    assert_equal(i.dtypes[1], np.dtype('f8'))
    i = nditer([array([3], dtype='u4'), array(0, dtype='i4')], ['common_dtype'], [['readonly', 'copy']] * 2, casting='safe')
    assert_equal(i.dtypes[0], np.dtype('i8'))
    assert_equal(i.dtypes[1], np.dtype('i8'))
    i = nditer([array([3], dtype='u4'), array(-12, dtype='i4')], ['common_dtype'], [['readonly', 'copy']] * 2, casting='safe')
    assert_equal(i.dtypes[0], np.dtype('i8'))
    assert_equal(i.dtypes[1], np.dtype('i8'))
    i = nditer([array([3], dtype='u4'), array(-12, dtype='i4'), array([2j], dtype='c8'), array([9], dtype='f8')], ['common_dtype'], [['readonly', 'copy']] * 4, casting='safe')
    assert_equal(i.dtypes[0], np.dtype('c16'))
    assert_equal(i.dtypes[1], np.dtype('c16'))
    assert_equal(i.dtypes[2], np.dtype('c16'))
    assert_equal(i.dtypes[3], np.dtype('c16'))
    assert_equal(i.value, (3, -12, 2j, 9))
    i = nditer([array([3], dtype='i4'), None, array([2j], dtype='c16')], [], [['readonly', 'copy'], ['writeonly', 'allocate'], ['writeonly']], casting='safe')
    assert_equal(i.dtypes[0], np.dtype('i4'))
    assert_equal(i.dtypes[1], np.dtype('i4'))
    assert_equal(i.dtypes[2], np.dtype('c16'))
    i = nditer([array([3], dtype='i4'), None, array([2j], dtype='c16')], ['common_dtype'], [['readonly', 'copy'], ['writeonly', 'allocate'], ['writeonly']], casting='safe')
    assert_equal(i.dtypes[0], np.dtype('c16'))
    assert_equal(i.dtypes[1], np.dtype('c16'))
    assert_equal(i.dtypes[2], np.dtype('c16'))

def test_iter_copy_if_overlap():
    if False:
        i = 10
        return i + 15
    for flag in ['readonly', 'writeonly', 'readwrite']:
        a = arange(10)
        i = nditer([a], ['copy_if_overlap'], [[flag]])
        with i:
            assert_(i.operands[0] is a)
    x = arange(10)
    a = x[1:]
    b = x[:-1]
    with nditer([a, b], ['copy_if_overlap'], [['readonly'], ['readwrite']]) as i:
        assert_(not np.shares_memory(*i.operands))
    x = arange(10)
    a = x
    b = x
    i = nditer([a, b], ['copy_if_overlap'], [['readonly', 'overlap_assume_elementwise'], ['readwrite', 'overlap_assume_elementwise']])
    with i:
        assert_(i.operands[0] is a and i.operands[1] is b)
    with nditer([a, b], ['copy_if_overlap'], [['readonly'], ['readwrite']]) as i:
        assert_(i.operands[0] is a and (not np.shares_memory(i.operands[1], b)))
    x = arange(10)
    a = x[::2]
    b = x[1::2]
    i = nditer([a, b], ['copy_if_overlap'], [['readonly'], ['writeonly']])
    assert_(i.operands[0] is a and i.operands[1] is b)
    x = arange(4, dtype=np.int8)
    a = x[3:]
    b = x.view(np.int32)[:1]
    with nditer([a, b], ['copy_if_overlap'], [['readonly'], ['writeonly']]) as i:
        assert_(not np.shares_memory(*i.operands))
    for flag in ['writeonly', 'readwrite']:
        x = np.ones([10, 10])
        a = x
        b = x.T
        c = x
        with nditer([a, b, c], ['copy_if_overlap'], [['readonly'], ['readonly'], [flag]]) as i:
            (a2, b2, c2) = i.operands
            assert_(not np.shares_memory(a2, c2))
            assert_(not np.shares_memory(b2, c2))
    x = np.ones([10, 10])
    a = x
    b = x.T
    c = x
    i = nditer([a, b, c], ['copy_if_overlap'], [['readonly'], ['readonly'], ['readonly']])
    (a2, b2, c2) = i.operands
    assert_(a is a2)
    assert_(b is b2)
    assert_(c is c2)
    x = np.ones([10, 10])
    a = x
    b = np.ones([10, 10])
    c = x.T
    i = nditer([a, b, c], ['copy_if_overlap'], [['readonly'], ['writeonly'], ['readonly']])
    (a2, b2, c2) = i.operands
    assert_(a is a2)
    assert_(b is b2)
    assert_(c is c2)
    x = np.arange(7)
    a = x[:3]
    b = x[3:6]
    c = x[4:7]
    i = nditer([a, b, c], ['copy_if_overlap'], [['readonly'], ['writeonly'], ['writeonly']])
    (a2, b2, c2) = i.operands
    assert_(a is a2)
    assert_(b is b2)
    assert_(c is c2)

def test_iter_op_axes():
    if False:
        print('Hello World!')
    a = arange(6).reshape(2, 3)
    i = nditer([a, a.T], [], [['readonly']] * 2, op_axes=[[0, 1], [1, 0]])
    assert_(all([x == y for (x, y) in i]))
    a = arange(24).reshape(2, 3, 4)
    i = nditer([a.T, a], [], [['readonly']] * 2, op_axes=[[2, 1, 0], None])
    assert_(all([x == y for (x, y) in i]))
    a = arange(1, 31).reshape(2, 3, 5)
    b = arange(1, 3)
    i = nditer([a, b], [], [['readonly']] * 2, op_axes=[None, [0, -1, -1]])
    assert_equal([x * y for (x, y) in i], (a * b.reshape(2, 1, 1)).ravel())
    b = arange(1, 4)
    i = nditer([a, b], [], [['readonly']] * 2, op_axes=[None, [-1, 0, -1]])
    assert_equal([x * y for (x, y) in i], (a * b.reshape(1, 3, 1)).ravel())
    b = arange(1, 6)
    i = nditer([a, b], [], [['readonly']] * 2, op_axes=[None, [np.newaxis, np.newaxis, 0]])
    assert_equal([x * y for (x, y) in i], (a * b.reshape(1, 1, 5)).ravel())
    a = arange(24).reshape(2, 3, 4)
    b = arange(40).reshape(5, 2, 4)
    i = nditer([a, b], ['multi_index'], [['readonly']] * 2, op_axes=[[0, 1, -1, -1], [-1, -1, 0, 1]])
    assert_equal(i.shape, (2, 3, 5, 2))
    a = arange(12).reshape(3, 4)
    b = arange(20).reshape(4, 5)
    i = nditer([a, b], ['multi_index'], [['readonly']] * 2, op_axes=[[0, -1], [-1, 1]])
    assert_equal(i.shape, (3, 5))

def test_iter_op_axes_errors():
    if False:
        return 10
    a = arange(6).reshape(2, 3)
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[0], [1], [0]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[2, 1], [0, 1]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[0, 1], [2, -1]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[0, 0], [0, 1]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[0, 1], [1, 1]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[0, 1], [0, 1, 0]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2, op_axes=[[0, 1], [1, 0]])

def test_iter_copy():
    if False:
        return 10
    a = arange(24).reshape(2, 3, 4)
    i = nditer(a)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])
    i.iterindex = 3
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])
    i = nditer(a, ['buffered', 'ranged'], order='F', buffersize=3)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])
    i.iterindex = 3
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])
    i.iterrange = (3, 9)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])
    i.iterrange = (2, 18)
    next(i)
    next(i)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])
    with nditer(a, ['buffered'], order='F', casting='unsafe', op_dtypes='f8', buffersize=5) as i:
        j = i.copy()
    assert_equal([x[()] for x in j], a.ravel(order='F'))
    a = arange(24, dtype='<i4').reshape(2, 3, 4)
    with nditer(a, ['buffered'], order='F', casting='unsafe', op_dtypes='>f8', buffersize=5) as i:
        j = i.copy()
    assert_equal([x[()] for x in j], a.ravel(order='F'))

@pytest.mark.parametrize('dtype', np.typecodes['All'])
@pytest.mark.parametrize('loop_dtype', np.typecodes['All'])
@pytest.mark.filterwarnings('ignore::numpy.exceptions.ComplexWarning')
def test_iter_copy_casts(dtype, loop_dtype):
    if False:
        while True:
            i = 10
    if loop_dtype.lower() == 'm':
        loop_dtype = loop_dtype + '[ms]'
    elif np.dtype(loop_dtype).itemsize == 0:
        loop_dtype = loop_dtype + '50'
    arr = np.ones(1000, dtype=np.dtype(dtype).newbyteorder())
    try:
        expected = arr.astype(loop_dtype)
    except Exception:
        return
    it = np.nditer((arr,), ['buffered', 'external_loop', 'refs_ok'], op_dtypes=[loop_dtype], casting='unsafe')
    if np.issubdtype(np.dtype(loop_dtype), np.number):
        assert_array_equal(expected, np.ones(1000, dtype=loop_dtype))
    it_copy = it.copy()
    res = next(it)
    del it
    res_copy = next(it_copy)
    del it_copy
    assert_array_equal(res, expected)
    assert_array_equal(res_copy, expected)

def test_iter_copy_casts_structured():
    if False:
        for i in range(10):
            print('nop')
    in_dtype = np.dtype([('a', np.dtype('i,')), ('b', np.dtype('>i,<i,>d,S17,>d,(3)f,O,i1'))])
    out_dtype = np.dtype([('a', np.dtype('O')), ('b', np.dtype('>i,>i,S17,>d,>U3,(3)d,i1,O'))])
    arr = np.ones(1000, dtype=in_dtype)
    it = np.nditer((arr,), ['buffered', 'external_loop', 'refs_ok'], op_dtypes=[out_dtype], casting='unsafe')
    it_copy = it.copy()
    res1 = next(it)
    del it
    res2 = next(it_copy)
    del it_copy
    expected = arr['a'].astype(out_dtype['a'])
    assert_array_equal(res1['a'], expected)
    assert_array_equal(res2['a'], expected)
    for field in in_dtype['b'].names:
        expected = arr['b'][field].astype(out_dtype['b'][field].base)
        assert_array_equal(res1['b'][field], expected)
        assert_array_equal(res2['b'][field], expected)

def test_iter_copy_casts_structured2():
    if False:
        i = 10
        return i + 15
    in_dtype = np.dtype([('a', np.dtype('O,O')), ('b', np.dtype('(5)O,(3)O,(1,)O,(1,)i,(1,)O'))])
    out_dtype = np.dtype([('a', np.dtype('O')), ('b', np.dtype('O,(3)i,(4)O,(4)O,(4)i'))])
    arr = np.ones(1, dtype=in_dtype)
    it = np.nditer((arr,), ['buffered', 'external_loop', 'refs_ok'], op_dtypes=[out_dtype], casting='unsafe')
    it_copy = it.copy()
    res1 = next(it)
    del it
    res2 = next(it_copy)
    del it_copy
    for res in (res1, res2):
        assert type(res['a'][0]) == tuple
        assert res['a'][0] == (1, 1)
    for res in (res1, res2):
        assert_array_equal(res['b']['f0'][0], np.ones(5, dtype=object))
        assert_array_equal(res['b']['f1'], np.ones((1, 3), dtype='i'))
        assert res['b']['f2'].shape == (1, 4)
        assert_array_equal(res['b']['f2'][0], np.ones(4, dtype=object))
        assert_array_equal(res['b']['f3'][0], np.ones(4, dtype=object))
        assert_array_equal(res['b']['f3'][0], np.ones(4, dtype='i'))

def test_iter_allocate_output_simple():
    if False:
        print('Hello World!')
    a = arange(6)
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']], op_dtypes=[None, np.dtype('f4')])
    assert_equal(i.operands[1].shape, a.shape)
    assert_equal(i.operands[1].dtype, np.dtype('f4'))

def test_iter_allocate_output_buffered_readwrite():
    if False:
        i = 10
        return i + 15
    a = arange(6)
    i = nditer([a, None], ['buffered', 'delay_bufalloc'], [['readonly'], ['allocate', 'readwrite']])
    with i:
        i.operands[1][:] = 1
        i.reset()
        for x in i:
            x[1][...] += x[0][...]
        assert_equal(i.operands[1], a + 1)

def test_iter_allocate_output_itorder():
    if False:
        print('Hello World!')
    a = arange(6, dtype='i4').reshape(2, 3)
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']], op_dtypes=[None, np.dtype('f4')])
    assert_equal(i.operands[1].shape, a.shape)
    assert_equal(i.operands[1].strides, a.strides)
    assert_equal(i.operands[1].dtype, np.dtype('f4'))
    a = arange(24, dtype='i4').reshape(2, 3, 4).T
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']], op_dtypes=[None, np.dtype('f4')])
    assert_equal(i.operands[1].shape, a.shape)
    assert_equal(i.operands[1].strides, a.strides)
    assert_equal(i.operands[1].dtype, np.dtype('f4'))
    a = arange(24, dtype='i4').reshape(2, 3, 4).swapaxes(0, 1)
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']], order='C', op_dtypes=[None, np.dtype('f4')])
    assert_equal(i.operands[1].shape, a.shape)
    assert_equal(i.operands[1].strides, (32, 16, 4))
    assert_equal(i.operands[1].dtype, np.dtype('f4'))

def test_iter_allocate_output_opaxes():
    if False:
        for i in range(10):
            print('nop')
    a = arange(24, dtype='i4').reshape(2, 3, 4)
    i = nditer([None, a], [], [['writeonly', 'allocate'], ['readonly']], op_dtypes=[np.dtype('u4'), None], op_axes=[[1, 2, 0], None])
    assert_equal(i.operands[0].shape, (4, 2, 3))
    assert_equal(i.operands[0].strides, (4, 48, 16))
    assert_equal(i.operands[0].dtype, np.dtype('u4'))

def test_iter_allocate_output_types_promotion():
    if False:
        for i in range(10):
            print('nop')
    i = nditer([array([3], dtype='f4'), array([0], dtype='f8'), None], [], [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('f8'))
    i = nditer([array([3], dtype='i4'), array([0], dtype='f4'), None], [], [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('f8'))
    i = nditer([array([3], dtype='f4'), array(0, dtype='f8'), None], [], [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('f8'))
    i = nditer([array([3], dtype='u4'), array(0, dtype='i4'), None], [], [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('i8'))
    i = nditer([array([3], dtype='u4'), array(-12, dtype='i4'), None], [], [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('i8'))

def test_iter_allocate_output_types_byte_order():
    if False:
        for i in range(10):
            print('nop')
    a = array([3], dtype='u4')
    a = a.view(a.dtype.newbyteorder())
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']])
    assert_equal(i.dtypes[0], i.dtypes[1])
    i = nditer([a, a, None], [], [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    assert_(i.dtypes[0] != i.dtypes[2])
    assert_equal(i.dtypes[0].newbyteorder('='), i.dtypes[2])

def test_iter_allocate_output_types_scalar():
    if False:
        while True:
            i = 10
    i = nditer([None, 1, 2.3, np.float32(12), np.complex128(3)], [], [['writeonly', 'allocate']] + [['readonly']] * 4)
    assert_equal(i.operands[0].dtype, np.dtype('complex128'))
    assert_equal(i.operands[0].ndim, 0)

def test_iter_allocate_output_subtype():
    if False:
        while True:
            i = 10

    class MyNDArray(np.ndarray):
        __array_priority__ = 15
    a = np.array([[1, 2], [3, 4]]).view(MyNDArray)
    b = np.arange(4).reshape(2, 2).T
    i = nditer([a, b, None], [], [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    assert_equal(type(a), type(i.operands[2]))
    assert_(type(b) is not type(i.operands[2]))
    assert_equal(i.operands[2].shape, (2, 2))
    i = nditer([a, b, None], [], [['readonly'], ['readonly'], ['writeonly', 'allocate', 'no_subtype']])
    assert_equal(type(b), type(i.operands[2]))
    assert_(type(a) is not type(i.operands[2]))
    assert_equal(i.operands[2].shape, (2, 2))

def test_iter_allocate_output_errors():
    if False:
        return 10
    a = arange(6)
    assert_raises(TypeError, nditer, [a, None], [], [['writeonly'], ['writeonly', 'allocate']])
    assert_raises(ValueError, nditer, [a, None], [], [['readonly'], ['allocate', 'readonly']])
    assert_raises(ValueError, nditer, [a, None], ['buffered'], ['allocate', 'readwrite'])
    assert_raises(TypeError, nditer, [None, None], [], [['writeonly', 'allocate'], ['writeonly', 'allocate']], op_dtypes=[None, np.dtype('f4')])
    a = arange(24, dtype='i4').reshape(2, 3, 4)
    assert_raises(ValueError, nditer, [a, None], [], [['readonly'], ['writeonly', 'allocate']], op_dtypes=[None, np.dtype('f4')], op_axes=[None, [0, np.newaxis, 1]])
    assert_raises(ValueError, nditer, [a, None], [], [['readonly'], ['writeonly', 'allocate']], op_dtypes=[None, np.dtype('f4')], op_axes=[None, [0, 3, 1]])
    assert_raises(ValueError, nditer, [a, None], [], [['readonly'], ['writeonly', 'allocate']], op_dtypes=[None, np.dtype('f4')], op_axes=[None, [0, 2, 1, 0]])
    a = arange(24, dtype='i4').reshape(2, 3, 4)
    assert_raises(ValueError, nditer, [a, None], ['reduce_ok'], [['readonly'], ['readwrite', 'allocate']], op_dtypes=[None, np.dtype('f4')], op_axes=[None, [0, np.newaxis, 2]])

def test_all_allocated():
    if False:
        i = 10
        return i + 15
    i = np.nditer([None], op_dtypes=['int64'])
    assert i.operands[0].shape == ()
    assert i.dtypes == (np.dtype('int64'),)
    i = np.nditer([None], op_dtypes=['int64'], itershape=(2, 3, 4))
    assert i.operands[0].shape == (2, 3, 4)

def test_iter_remove_axis():
    if False:
        return 10
    a = arange(24).reshape(2, 3, 4)
    i = nditer(a, ['multi_index'])
    i.remove_axis(1)
    assert_equal([x for x in i], a[:, 0, :].ravel())
    a = a[::-1, :, :]
    i = nditer(a, ['multi_index'])
    i.remove_axis(0)
    assert_equal([x for x in i], a[0, :, :].ravel())

def test_iter_remove_multi_index_inner_loop():
    if False:
        return 10
    a = arange(24).reshape(2, 3, 4)
    i = nditer(a, ['multi_index'])
    assert_equal(i.ndim, 3)
    assert_equal(i.shape, (2, 3, 4))
    assert_equal(i.itviews[0].shape, (2, 3, 4))
    before = [x for x in i]
    i.remove_multi_index()
    after = [x for x in i]
    assert_equal(before, after)
    assert_equal(i.ndim, 1)
    assert_raises(ValueError, lambda i: i.shape, i)
    assert_equal(i.itviews[0].shape, (24,))
    i.reset()
    assert_equal(i.itersize, 24)
    assert_equal(i[0].shape, tuple())
    i.enable_external_loop()
    assert_equal(i.itersize, 24)
    assert_equal(i[0].shape, (24,))
    assert_equal(i.value, arange(24))

def test_iter_iterindex():
    if False:
        for i in range(10):
            print('nop')
    buffersize = 5
    a = arange(24).reshape(4, 3, 2)
    for flags in ([], ['buffered']):
        i = nditer(a, flags, buffersize=buffersize)
        assert_equal(iter_iterindices(i), list(range(24)))
        i.iterindex = 2
        assert_equal(iter_iterindices(i), list(range(2, 24)))
        i = nditer(a, flags, order='F', buffersize=buffersize)
        assert_equal(iter_iterindices(i), list(range(24)))
        i.iterindex = 5
        assert_equal(iter_iterindices(i), list(range(5, 24)))
        i = nditer(a[::-1], flags, order='F', buffersize=buffersize)
        assert_equal(iter_iterindices(i), list(range(24)))
        i.iterindex = 9
        assert_equal(iter_iterindices(i), list(range(9, 24)))
        i = nditer(a[::-1, ::-1], flags, order='C', buffersize=buffersize)
        assert_equal(iter_iterindices(i), list(range(24)))
        i.iterindex = 13
        assert_equal(iter_iterindices(i), list(range(13, 24)))
        i = nditer(a[::1, ::-1], flags, buffersize=buffersize)
        assert_equal(iter_iterindices(i), list(range(24)))
        i.iterindex = 23
        assert_equal(iter_iterindices(i), list(range(23, 24)))
        i.reset()
        i.iterindex = 2
        assert_equal(iter_iterindices(i), list(range(2, 24)))

def test_iter_iterrange():
    if False:
        i = 10
        return i + 15
    buffersize = 5
    a = arange(24, dtype='i4').reshape(4, 3, 2)
    a_fort = a.ravel(order='F')
    i = nditer(a, ['ranged'], ['readonly'], order='F', buffersize=buffersize)
    assert_equal(i.iterrange, (0, 24))
    assert_equal([x[()] for x in i], a_fort)
    for r in [(0, 24), (1, 2), (3, 24), (5, 5), (0, 20), (23, 24)]:
        i.iterrange = r
        assert_equal(i.iterrange, r)
        assert_equal([x[()] for x in i], a_fort[r[0]:r[1]])
    i = nditer(a, ['ranged', 'buffered'], ['readonly'], order='F', op_dtypes='f8', buffersize=buffersize)
    assert_equal(i.iterrange, (0, 24))
    assert_equal([x[()] for x in i], a_fort)
    for r in [(0, 24), (1, 2), (3, 24), (5, 5), (0, 20), (23, 24)]:
        i.iterrange = r
        assert_equal(i.iterrange, r)
        assert_equal([x[()] for x in i], a_fort[r[0]:r[1]])

    def get_array(i):
        if False:
            print('Hello World!')
        val = np.array([], dtype='f8')
        for x in i:
            val = np.concatenate((val, x))
        return val
    i = nditer(a, ['ranged', 'buffered', 'external_loop'], ['readonly'], order='F', op_dtypes='f8', buffersize=buffersize)
    assert_equal(i.iterrange, (0, 24))
    assert_equal(get_array(i), a_fort)
    for r in [(0, 24), (1, 2), (3, 24), (5, 5), (0, 20), (23, 24)]:
        i.iterrange = r
        assert_equal(i.iterrange, r)
        assert_equal(get_array(i), a_fort[r[0]:r[1]])

def test_iter_buffering():
    if False:
        i = 10
        return i + 15
    arrays = []
    _tmp = np.arange(24, dtype='c16').reshape(2, 3, 4).T
    _tmp = _tmp.view(_tmp.dtype.newbyteorder()).byteswap()
    arrays.append(_tmp)
    arrays.append(np.arange(10, dtype='f4'))
    a = np.zeros((4 * 16 + 1,), dtype='i1')[1:]
    a.dtype = 'i4'
    a[:] = np.arange(16, dtype='i4')
    arrays.append(a)
    arrays.append(np.arange(120, dtype='i4').reshape(5, 3, 2, 4).T)
    for a in arrays:
        for buffersize in (1, 2, 3, 5, 8, 11, 16, 1024):
            vals = []
            i = nditer(a, ['buffered', 'external_loop'], [['readonly', 'nbo', 'aligned']], order='C', casting='equiv', buffersize=buffersize)
            while not i.finished:
                assert_(i[0].size <= buffersize)
                vals.append(i[0].copy())
                i.iternext()
            assert_equal(np.concatenate(vals), a.ravel(order='C'))

def test_iter_write_buffering():
    if False:
        while True:
            i = 10
    a = np.arange(24).reshape(2, 3, 4).T
    a = a.view(a.dtype.newbyteorder()).byteswap()
    i = nditer(a, ['buffered'], [['readwrite', 'nbo', 'aligned']], casting='equiv', order='C', buffersize=16)
    x = 0
    with i:
        while not i.finished:
            i[0] = x
            x += 1
            i.iternext()
    assert_equal(a.ravel(order='C'), np.arange(24))

def test_iter_buffering_delayed_alloc():
    if False:
        i = 10
        return i + 15
    a = np.arange(6)
    b = np.arange(1, dtype='f4')
    i = nditer([a, b], ['buffered', 'delay_bufalloc', 'multi_index', 'reduce_ok'], ['readwrite'], casting='unsafe', op_dtypes='f4')
    assert_(i.has_delayed_bufalloc)
    assert_raises(ValueError, lambda i: i.multi_index, i)
    assert_raises(ValueError, lambda i: i[0], i)
    assert_raises(ValueError, lambda i: i[0:2], i)

    def assign_iter(i):
        if False:
            return 10
        i[0] = 0
    assert_raises(ValueError, assign_iter, i)
    i.reset()
    assert_(not i.has_delayed_bufalloc)
    assert_equal(i.multi_index, (0,))
    with i:
        assert_equal(i[0], 0)
        i[1] = 1
        assert_equal(i[0:2], [0, 1])
        assert_equal([[x[0][()], x[1][()]] for x in i], list(zip(range(6), [1] * 6)))

def test_iter_buffered_cast_simple():
    if False:
        for i in range(10):
            print('nop')
    a = np.arange(10, dtype='f4')
    i = nditer(a, ['buffered', 'external_loop'], [['readwrite', 'nbo', 'aligned']], casting='same_kind', op_dtypes=[np.dtype('f8')], buffersize=3)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype='f4'))

def test_iter_buffered_cast_byteswapped():
    if False:
        print('Hello World!')
    a = np.arange(10, dtype='f4')
    a = a.view(a.dtype.newbyteorder()).byteswap()
    i = nditer(a, ['buffered', 'external_loop'], [['readwrite', 'nbo', 'aligned']], casting='same_kind', op_dtypes=[np.dtype('f8').newbyteorder()], buffersize=3)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype='f4'))
    with suppress_warnings() as sup:
        sup.filter(np.exceptions.ComplexWarning)
        a = np.arange(10, dtype='f8')
        a = a.view(a.dtype.newbyteorder()).byteswap()
        i = nditer(a, ['buffered', 'external_loop'], [['readwrite', 'nbo', 'aligned']], casting='unsafe', op_dtypes=[np.dtype('c8').newbyteorder()], buffersize=3)
        with i:
            for v in i:
                v[...] *= 2
        assert_equal(a, 2 * np.arange(10, dtype='f8'))

def test_iter_buffered_cast_byteswapped_complex():
    if False:
        return 10
    a = np.arange(10, dtype='c8')
    a = a.view(a.dtype.newbyteorder()).byteswap()
    a += 2j
    i = nditer(a, ['buffered', 'external_loop'], [['readwrite', 'nbo', 'aligned']], casting='same_kind', op_dtypes=[np.dtype('c16')], buffersize=3)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype='c8') + 4j)
    a = np.arange(10, dtype='c8')
    a += 2j
    i = nditer(a, ['buffered', 'external_loop'], [['readwrite', 'nbo', 'aligned']], casting='same_kind', op_dtypes=[np.dtype('c16').newbyteorder()], buffersize=3)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype='c8') + 4j)
    a = np.arange(10, dtype=np.clongdouble)
    a = a.view(a.dtype.newbyteorder()).byteswap()
    a += 2j
    i = nditer(a, ['buffered', 'external_loop'], [['readwrite', 'nbo', 'aligned']], casting='same_kind', op_dtypes=[np.dtype('c16')], buffersize=3)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype=np.clongdouble) + 4j)
    a = np.arange(10, dtype=np.longdouble)
    a = a.view(a.dtype.newbyteorder()).byteswap()
    i = nditer(a, ['buffered', 'external_loop'], [['readwrite', 'nbo', 'aligned']], casting='same_kind', op_dtypes=[np.dtype('f4')], buffersize=7)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype=np.longdouble))

def test_iter_buffered_cast_structured_type():
    if False:
        print('Hello World!')
    sdt = [('a', 'f4'), ('b', 'i8'), ('c', 'c8', (2, 3)), ('d', 'O')]
    a = np.arange(3, dtype='f4') + 0.5
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt)
    vals = [np.array(x) for x in i]
    assert_equal(vals[0]['a'], 0.5)
    assert_equal(vals[0]['b'], 0)
    assert_equal(vals[0]['c'], [[0.5] * 3] * 2)
    assert_equal(vals[0]['d'], 0.5)
    assert_equal(vals[1]['a'], 1.5)
    assert_equal(vals[1]['b'], 1)
    assert_equal(vals[1]['c'], [[1.5] * 3] * 2)
    assert_equal(vals[1]['d'], 1.5)
    assert_equal(vals[0].dtype, np.dtype(sdt))
    sdt = [('a', 'f4'), ('b', 'i8'), ('c', 'c8', (2, 3)), ('d', 'O')]
    a = np.zeros((3,), dtype='O')
    a[0] = (0.5, 0.5, [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], 0.5)
    a[1] = (1.5, 1.5, [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]], 1.5)
    a[2] = (2.5, 2.5, [[2.5, 2.5, 2.5], [2.5, 2.5, 2.5]], 2.5)
    if HAS_REFCOUNT:
        rc = sys.getrefcount(a[0])
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt)
    vals = [x.copy() for x in i]
    assert_equal(vals[0]['a'], 0.5)
    assert_equal(vals[0]['b'], 0)
    assert_equal(vals[0]['c'], [[0.5] * 3] * 2)
    assert_equal(vals[0]['d'], 0.5)
    assert_equal(vals[1]['a'], 1.5)
    assert_equal(vals[1]['b'], 1)
    assert_equal(vals[1]['c'], [[1.5] * 3] * 2)
    assert_equal(vals[1]['d'], 1.5)
    assert_equal(vals[0].dtype, np.dtype(sdt))
    (vals, i, x) = [None] * 3
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(a[0]), rc)
    sdt = [('a', 'f4')]
    a = np.array([(5.5,), (8,)], dtype=sdt)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes='i4')
    assert_equal([x_[()] for x_ in i], [5, 8])
    sdt = [('a', 'f4'), ('b', 'i8'), ('d', 'O')]
    a = np.array([(5.5, 7, 'test'), (8, 10, 11)], dtype=sdt)
    assert_raises(TypeError, lambda : nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes='i4'))
    sdt1 = [('a', 'f4'), ('b', 'i8'), ('d', 'O')]
    sdt2 = [('d', 'u2'), ('a', 'O'), ('b', 'f8')]
    a = np.array([(1, 2, 3), (4, 5, 6)], dtype=sdt1)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    assert_equal([np.array(x_) for x_ in i], [np.array((1, 2, 3), dtype=sdt2), np.array((4, 5, 6), dtype=sdt2)])

def test_iter_buffered_cast_structured_type_failure_with_cleanup():
    if False:
        return 10
    sdt1 = [('a', 'f4'), ('b', 'i8'), ('d', 'O')]
    sdt2 = [('b', 'O'), ('a', 'f8')]
    a = np.array([(1, 2, 3), (4, 5, 6)], dtype=sdt1)
    for intent in ['readwrite', 'readonly', 'writeonly']:
        simple_arr = np.array([1, 2], dtype='i,i')
        with pytest.raises(TypeError):
            nditer((simple_arr, a), ['buffered', 'refs_ok'], [intent, intent], casting='unsafe', op_dtypes=['f,f', sdt2])

def test_buffered_cast_error_paths():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):
        np.nditer((np.array('a', dtype='S1'),), op_dtypes=['i'], casting='unsafe', flags=['buffered'])
    it = np.nditer((np.array(1, dtype='i'),), op_dtypes=['S1'], op_flags=['writeonly'], casting='unsafe', flags=['buffered'])
    with pytest.raises(ValueError):
        with it:
            buf = next(it)
            buf[...] = 'a'

@pytest.mark.skipif(IS_WASM, reason='Cannot start subprocess')
@pytest.mark.skipif(not HAS_REFCOUNT, reason='PyPy seems to not hit this.')
def test_buffered_cast_error_paths_unraisable():
    if False:
        print('Hello World!')
    code = textwrap.dedent('\n        import numpy as np\n    \n        it = np.nditer((np.array(1, dtype="i"),), op_dtypes=["S1"],\n                       op_flags=["writeonly"], casting="unsafe", flags=["buffered"])\n        buf = next(it)\n        buf[...] = "a"\n        del buf, it  # Flushing only happens during deallocate right now.\n        ')
    res = subprocess.check_output([sys.executable, '-c', code], stderr=subprocess.STDOUT, text=True)
    assert 'ValueError' in res

def test_iter_buffered_cast_subarray():
    if False:
        for i in range(10):
            print('nop')
    sdt1 = [('a', 'f4')]
    sdt2 = [('a', 'f8', (3, 2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    for (x, count) in zip(i, list(range(6))):
        assert_(np.all(x['a'] == count))
    sdt1 = [('a', 'O', (1, 1))]
    sdt2 = [('a', 'O', (3, 2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'][:, 0, 0] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readwrite'], casting='unsafe', op_dtypes=sdt2)
    with i:
        assert_equal(i[0].dtype, np.dtype(sdt2))
        count = 0
        for x in i:
            assert_(np.all(x['a'] == count))
            x['a'][0] += 2
            count += 1
    assert_equal(a['a'], np.arange(6).reshape(6, 1, 1) + 2)
    sdt1 = [('a', 'O', (3, 2, 2))]
    sdt2 = [('a', 'O', (1,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'][:, 0, 0, 0] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readwrite'], casting='unsafe', op_dtypes=sdt2)
    with i:
        assert_equal(i[0].dtype, np.dtype(sdt2))
        count = 0
        for x in i:
            assert_equal(x['a'], count)
            x['a'] += 2
            count += 1
    assert_equal(a['a'], np.arange(6).reshape(6, 1, 1, 1) * np.ones((1, 3, 2, 2)) + 2)
    sdt1 = [('a', 'f8', (3, 2, 2))]
    sdt2 = [('a', 'O', (1,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'][:, 0, 0, 0] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'], count)
        count += 1
    sdt1 = [('a', 'O', (3, 2, 2))]
    sdt2 = [('a', 'f4', (1,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'][:, 0, 0, 0] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'], count)
        count += 1
    sdt1 = [('a', 'O', (3, 2, 2))]
    sdt2 = [('a', 'f4', (3, 2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 3 * 2 * 2).reshape(6, 3, 2, 2)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'], a[count]['a'])
        count += 1
    sdt1 = [('a', 'f8', (6,))]
    sdt2 = [('a', 'f4', (2,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 6).reshape(6, 6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'], a[count]['a'][:2])
        count += 1
    sdt1 = [('a', 'f8', (2,))]
    sdt2 = [('a', 'f4', (6,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 2).reshape(6, 2)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'][:2], a[count]['a'])
        assert_equal(x['a'][2:], [0, 0, 0, 0])
        count += 1
    sdt1 = [('a', 'f8', (2,))]
    sdt2 = [('a', 'f4', (2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 2).reshape(6, 2)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'][0], a[count]['a'])
        assert_equal(x['a'][1], a[count]['a'])
        count += 1
    sdt1 = [('a', 'f8', (2, 1))]
    sdt2 = [('a', 'f4', (3, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 2).reshape(6, 2, 1)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'][:2, 0], a[count]['a'][:, 0])
        assert_equal(x['a'][:2, 1], a[count]['a'][:, 0])
        assert_equal(x['a'][2, :], [0, 0])
        count += 1
    sdt1 = [('a', 'f8', (2, 3))]
    sdt2 = [('a', 'f4', (3, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 2 * 3).reshape(6, 2, 3)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'][:2, 0], a[count]['a'][:, 0])
        assert_equal(x['a'][:2, 1], a[count]['a'][:, 1])
        assert_equal(x['a'][2, :], [0, 0])
        count += 1

def test_iter_buffering_badwriteback():
    if False:
        return 10
    a = np.arange(6).reshape(2, 3, 1)
    b = np.arange(12).reshape(2, 3, 2)
    assert_raises(ValueError, nditer, [a, b], ['buffered', 'external_loop'], [['readwrite'], ['writeonly']], order='C')
    nditer([a, b], ['buffered', 'external_loop'], [['readonly'], ['writeonly']], order='C')
    a = np.arange(1).reshape(1, 1, 1)
    nditer([a, b], ['buffered', 'external_loop', 'reduce_ok'], [['readwrite'], ['writeonly']], order='C')
    a = np.arange(6).reshape(1, 3, 2)
    assert_raises(ValueError, nditer, [a, b], ['buffered', 'external_loop'], [['readwrite'], ['writeonly']], order='C')
    a = np.arange(4).reshape(2, 1, 2)
    assert_raises(ValueError, nditer, [a, b], ['buffered', 'external_loop'], [['readwrite'], ['writeonly']], order='C')

def test_iter_buffering_string():
    if False:
        for i in range(10):
            print('nop')
    a = np.array(['abc', 'a', 'abcd'], dtype=np.bytes_)
    assert_equal(a.dtype, np.dtype('S4'))
    assert_raises(TypeError, nditer, a, ['buffered'], ['readonly'], op_dtypes='S2')
    i = nditer(a, ['buffered'], ['readonly'], op_dtypes='S6')
    assert_equal(i[0], b'abc')
    assert_equal(i[0].dtype, np.dtype('S6'))
    a = np.array(['abc', 'a', 'abcd'], dtype=np.str_)
    assert_equal(a.dtype, np.dtype('U4'))
    assert_raises(TypeError, nditer, a, ['buffered'], ['readonly'], op_dtypes='U2')
    i = nditer(a, ['buffered'], ['readonly'], op_dtypes='U6')
    assert_equal(i[0], 'abc')
    assert_equal(i[0].dtype, np.dtype('U6'))

def test_iter_buffering_growinner():
    if False:
        i = 10
        return i + 15
    a = np.arange(30)
    i = nditer(a, ['buffered', 'growinner', 'external_loop'], buffersize=5)
    assert_equal(i[0].size, a.size)

@pytest.mark.slow
def test_iter_buffered_reduce_reuse():
    if False:
        print('Hello World!')
    a = np.arange(2 * 3 ** 5)[3 ** 5:3 ** 5 + 1]
    flags = ['buffered', 'delay_bufalloc', 'multi_index', 'reduce_ok', 'refs_ok']
    op_flags = [('readonly',), ('readwrite', 'allocate')]
    op_axes_list = [[(0, 1, 2), (0, 1, -1)], [(0, 1, 2), (0, -1, -1)]]
    op_dtypes = [float, a.dtype]

    def get_params():
        if False:
            for i in range(10):
                print('nop')
        for xs in range(-3 ** 2, 3 ** 2 + 1):
            for ys in range(xs, 3 ** 2 + 1):
                for op_axes in op_axes_list:
                    strides = (xs * a.itemsize, ys * a.itemsize, a.itemsize)
                    arr = np.lib.stride_tricks.as_strided(a, (3, 3, 3), strides)
                    for skip in [0, 1]:
                        yield (arr, op_axes, skip)
    for (arr, op_axes, skip) in get_params():
        nditer2 = np.nditer([arr.copy(), None], op_axes=op_axes, flags=flags, op_flags=op_flags, op_dtypes=op_dtypes)
        with nditer2:
            nditer2.operands[-1][...] = 0
            nditer2.reset()
            nditer2.iterindex = skip
            for (a2_in, b2_in) in nditer2:
                b2_in += a2_in.astype(np.int_)
            comp_res = nditer2.operands[-1]
        for bufsize in range(0, 3 ** 3):
            nditer1 = np.nditer([arr, None], op_axes=op_axes, flags=flags, op_flags=op_flags, buffersize=bufsize, op_dtypes=op_dtypes)
            with nditer1:
                nditer1.operands[-1][...] = 0
                nditer1.reset()
                nditer1.iterindex = skip
                for (a1_in, b1_in) in nditer1:
                    b1_in += a1_in.astype(np.int_)
                res = nditer1.operands[-1]
            assert_array_equal(res, comp_res)

def test_iter_no_broadcast():
    if False:
        return 10
    a = np.arange(24).reshape(2, 3, 4)
    b = np.arange(6).reshape(2, 3, 1)
    c = np.arange(12).reshape(3, 4)
    nditer([a, b, c], [], [['readonly', 'no_broadcast'], ['readonly'], ['readonly']])
    assert_raises(ValueError, nditer, [a, b, c], [], [['readonly'], ['readonly', 'no_broadcast'], ['readonly']])
    assert_raises(ValueError, nditer, [a, b, c], [], [['readonly'], ['readonly'], ['readonly', 'no_broadcast']])

class TestIterNested:

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        a = arange(12).reshape(2, 3, 2)
        (i, j) = np.nested_iters(a, [[0], [1, 2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        (i, j) = np.nested_iters(a, [[0, 1], [2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
        (i, j) = np.nested_iters(a, [[0, 2], [1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])

    def test_reorder(self):
        if False:
            for i in range(10):
                print('nop')
        a = arange(12).reshape(2, 3, 2)
        (i, j) = np.nested_iters(a, [[0], [2, 1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        (i, j) = np.nested_iters(a, [[1, 0], [2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
        (i, j) = np.nested_iters(a, [[2, 0], [1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])
        (i, j) = np.nested_iters(a, [[0], [2, 1]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4, 1, 3, 5], [6, 8, 10, 7, 9, 11]])
        (i, j) = np.nested_iters(a, [[1, 0], [2]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1], [6, 7], [2, 3], [8, 9], [4, 5], [10, 11]])
        (i, j) = np.nested_iters(a, [[2, 0], [1]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4], [6, 8, 10], [1, 3, 5], [7, 9, 11]])

    def test_flip_axes(self):
        if False:
            i = 10
            return i + 15
        a = arange(12).reshape(2, 3, 2)[::-1, ::-1, ::-1]
        (i, j) = np.nested_iters(a, [[0], [1, 2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        (i, j) = np.nested_iters(a, [[0, 1], [2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
        (i, j) = np.nested_iters(a, [[0, 2], [1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])
        (i, j) = np.nested_iters(a, [[0], [1, 2]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[11, 10, 9, 8, 7, 6], [5, 4, 3, 2, 1, 0]])
        (i, j) = np.nested_iters(a, [[0, 1], [2]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[11, 10], [9, 8], [7, 6], [5, 4], [3, 2], [1, 0]])
        (i, j) = np.nested_iters(a, [[0, 2], [1]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[11, 9, 7], [10, 8, 6], [5, 3, 1], [4, 2, 0]])

    def test_broadcast(self):
        if False:
            i = 10
            return i + 15
        a = arange(2).reshape(2, 1)
        b = arange(3).reshape(1, 3)
        (i, j) = np.nested_iters([a, b], [[0], [1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]]])
        (i, j) = np.nested_iters([a, b], [[1], [0]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[[0, 0], [1, 0]], [[0, 1], [1, 1]], [[0, 2], [1, 2]]])

    def test_dtype_copy(self):
        if False:
            for i in range(10):
                print('nop')
        a = arange(6, dtype='i4').reshape(2, 3)
        (i, j) = np.nested_iters(a, [[0], [1]], op_flags=['readonly', 'copy'], op_dtypes='f8')
        assert_equal(j[0].dtype, np.dtype('f8'))
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1, 2], [3, 4, 5]])
        vals = None
        a = arange(6, dtype='f4').reshape(2, 3)
        (i, j) = np.nested_iters(a, [[0], [1]], op_flags=['readwrite', 'updateifcopy'], casting='same_kind', op_dtypes='f8')
        with i, j:
            assert_equal(j[0].dtype, np.dtype('f8'))
            for x in i:
                for y in j:
                    y[...] += 1
            assert_equal(a, [[0, 1, 2], [3, 4, 5]])
        assert_equal(a, [[1, 2, 3], [4, 5, 6]])
        a = arange(6, dtype='f4').reshape(2, 3)
        (i, j) = np.nested_iters(a, [[0], [1]], op_flags=['readwrite', 'updateifcopy'], casting='same_kind', op_dtypes='f8')
        assert_equal(j[0].dtype, np.dtype('f8'))
        for x in i:
            for y in j:
                y[...] += 1
        assert_equal(a, [[0, 1, 2], [3, 4, 5]])
        i.close()
        j.close()
        assert_equal(a, [[1, 2, 3], [4, 5, 6]])

    def test_dtype_buffered(self):
        if False:
            return 10
        a = arange(6, dtype='f4').reshape(2, 3)
        (i, j) = np.nested_iters(a, [[0], [1]], flags=['buffered'], op_flags=['readwrite'], casting='same_kind', op_dtypes='f8')
        assert_equal(j[0].dtype, np.dtype('f8'))
        for x in i:
            for y in j:
                y[...] += 1
        assert_equal(a, [[1, 2, 3], [4, 5, 6]])

    def test_0d(self):
        if False:
            i = 10
            return i + 15
        a = np.arange(12).reshape(2, 3, 2)
        (i, j) = np.nested_iters(a, [[], [1, 0, 2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
        (i, j) = np.nested_iters(a, [[1, 0, 2], []])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]])
        (i, j, k) = np.nested_iters(a, [[2, 0], [], [1]])
        vals = []
        for x in i:
            for y in j:
                vals.append([z for z in k])
        assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])

    def test_iter_nested_iters_dtype_buffered(self):
        if False:
            print('Hello World!')
        a = arange(6, dtype='f4').reshape(2, 3)
        (i, j) = np.nested_iters(a, [[0], [1]], flags=['buffered'], op_flags=['readwrite'], casting='same_kind', op_dtypes='f8')
        with i, j:
            assert_equal(j[0].dtype, np.dtype('f8'))
            for x in i:
                for y in j:
                    y[...] += 1
        assert_equal(a, [[1, 2, 3], [4, 5, 6]])

def test_iter_reduction_error():
    if False:
        while True:
            i = 10
    a = np.arange(6)
    assert_raises(ValueError, nditer, [a, None], [], [['readonly'], ['readwrite', 'allocate']], op_axes=[[0], [-1]])
    a = np.arange(6).reshape(2, 3)
    assert_raises(ValueError, nditer, [a, None], ['external_loop'], [['readonly'], ['readwrite', 'allocate']], op_axes=[[0, 1], [-1, -1]])

def test_iter_reduction():
    if False:
        return 10
    a = np.arange(6)
    i = nditer([a, None], ['reduce_ok'], [['readonly'], ['readwrite', 'allocate']], op_axes=[[0], [-1]])
    with i:
        i.operands[1][...] = 0
        for (x, y) in i:
            y[...] += x
        assert_equal(i.operands[1].ndim, 0)
        assert_equal(i.operands[1], np.sum(a))
    a = np.arange(6).reshape(2, 3)
    i = nditer([a, None], ['reduce_ok', 'external_loop'], [['readonly'], ['readwrite', 'allocate']], op_axes=[[0, 1], [-1, -1]])
    with i:
        i.operands[1][...] = 0
        assert_equal(i[1].shape, (6,))
        assert_equal(i[1].strides, (0,))
        for (x, y) in i:
            for j in range(len(y)):
                y[j] += x[j]
        assert_equal(i.operands[1].ndim, 0)
        assert_equal(i.operands[1], np.sum(a))
    a = np.ones((2, 3, 5))
    it1 = nditer([a, None], ['reduce_ok', 'external_loop'], [['readonly'], ['readwrite', 'allocate']], op_axes=[None, [0, -1, 1]])
    it2 = nditer([a, None], ['reduce_ok', 'external_loop', 'buffered', 'delay_bufalloc'], [['readonly'], ['readwrite', 'allocate']], op_axes=[None, [0, -1, 1]], buffersize=10)
    with it1, it2:
        it1.operands[1].fill(0)
        it2.operands[1].fill(0)
        it2.reset()
        for x in it1:
            x[1][...] += x[0]
        for x in it2:
            x[1][...] += x[0]
        assert_equal(it1.operands[1], it2.operands[1])
        assert_equal(it2.operands[1].sum(), a.size)

def test_iter_buffering_reduction():
    if False:
        for i in range(10):
            print('nop')
    a = np.arange(6)
    b = np.array(0.0, dtype='f8').byteswap()
    b = b.view(b.dtype.newbyteorder())
    i = nditer([a, b], ['reduce_ok', 'buffered'], [['readonly'], ['readwrite', 'nbo']], op_axes=[[0], [-1]])
    with i:
        assert_equal(i[1].dtype, np.dtype('f8'))
        assert_(i[1].dtype != b.dtype)
        for (x, y) in i:
            y[...] += x
    assert_equal(b, np.sum(a))
    a = np.arange(6).reshape(2, 3)
    b = np.array([0, 0], dtype='f8').byteswap()
    b = b.view(b.dtype.newbyteorder())
    i = nditer([a, b], ['reduce_ok', 'external_loop', 'buffered'], [['readonly'], ['readwrite', 'nbo']], op_axes=[[0, 1], [0, -1]])
    with i:
        assert_equal(i[1].shape, (3,))
        assert_equal(i[1].strides, (0,))
        for (x, y) in i:
            for j in range(len(y)):
                y[j] += x[j]
    assert_equal(b, np.sum(a, axis=1))
    p = np.arange(2) + 1
    it = np.nditer([p, None], ['delay_bufalloc', 'reduce_ok', 'buffered', 'external_loop'], [['readonly'], ['readwrite', 'allocate']], op_axes=[[-1, 0], [-1, -1]], itershape=(2, 2))
    with it:
        it.operands[1].fill(0)
        it.reset()
        assert_equal(it[0], [1, 2, 1, 2])
    x = np.ones((7, 13, 8), np.int8)[4:6, 1:11:6, 1:5].transpose(1, 2, 0)
    x[...] = np.arange(x.size).reshape(x.shape)
    y_base = np.arange(4 * 4, dtype=np.int8).reshape(4, 4)
    y_base_copy = y_base.copy()
    y = y_base[::2, :, None]
    it = np.nditer([y, x], ['buffered', 'external_loop', 'reduce_ok'], [['readwrite'], ['readonly']])
    with it:
        for (a, b) in it:
            a.fill(2)
    assert_equal(y_base[1::2], y_base_copy[1::2])
    assert_equal(y_base[::2], 2)

def test_iter_buffering_reduction_reuse_reduce_loops():
    if False:
        i = 10
        return i + 15
    a = np.zeros((2, 7))
    b = np.zeros((1, 7))
    it = np.nditer([a, b], flags=['reduce_ok', 'external_loop', 'buffered'], op_flags=[['readonly'], ['readwrite']], buffersize=5)
    with it:
        bufsizes = [x.shape[0] for (x, y) in it]
    assert_equal(bufsizes, [5, 2, 5, 2])
    assert_equal(sum(bufsizes), a.size)

def test_iter_writemasked_badinput():
    if False:
        while True:
            i = 10
    a = np.zeros((2, 3))
    b = np.zeros((3,))
    m = np.array([[True, True, False], [False, True, False]])
    m2 = np.array([True, True, False])
    m3 = np.array([0, 1, 1], dtype='u1')
    mbad1 = np.array([0, 1, 1], dtype='i1')
    mbad2 = np.array([0, 1, 1], dtype='f4')
    assert_raises(ValueError, nditer, [a, m], [], [['readwrite', 'writemasked'], ['readonly']])
    assert_raises(ValueError, nditer, [a, m], [], [['readonly', 'writemasked'], ['readonly', 'arraymask']])
    assert_raises(ValueError, nditer, [a, m], [], [['readonly'], ['readwrite', 'arraymask', 'writemasked']])
    assert_raises(ValueError, nditer, [a, m, m2], [], [['readwrite', 'writemasked'], ['readonly', 'arraymask'], ['readonly', 'arraymask']])
    assert_raises(ValueError, nditer, [a, m], [], [['readwrite'], ['readonly', 'arraymask']])
    assert_raises(ValueError, nditer, [a, b, m], ['reduce_ok'], [['readonly'], ['readwrite', 'writemasked'], ['readonly', 'arraymask']])
    np.nditer([a, b, m2], ['reduce_ok'], [['readonly'], ['readwrite', 'writemasked'], ['readonly', 'arraymask']])
    assert_raises(ValueError, nditer, [a, b, m2], ['reduce_ok'], [['readonly'], ['readwrite', 'writemasked'], ['readwrite', 'arraymask']])
    np.nditer([a, m3], ['buffered'], [['readwrite', 'writemasked'], ['readonly', 'arraymask']], op_dtypes=['f4', None], casting='same_kind')
    assert_raises(TypeError, np.nditer, [a, mbad1], ['buffered'], [['readwrite', 'writemasked'], ['readonly', 'arraymask']], op_dtypes=['f4', None], casting='same_kind')
    assert_raises(TypeError, np.nditer, [a, mbad2], ['buffered'], [['readwrite', 'writemasked'], ['readonly', 'arraymask']], op_dtypes=['f4', None], casting='same_kind')

def _is_buffered(iterator):
    if False:
        return 10
    try:
        iterator.itviews
    except ValueError:
        return True
    return False

@pytest.mark.parametrize('a', [np.zeros((3,), dtype='f8'), np.zeros((9876, 3 * 5), dtype='f8')[::2, :], np.zeros((4, 312, 124, 3), dtype='f8')[::2, :, ::2, :], np.zeros((9,), dtype='f8')[::3], np.zeros((9876, 3 * 10), dtype='f8')[::2, ::5], np.zeros((4, 312, 124, 3), dtype='f8')[::2, :, ::2, ::-1]])
def test_iter_writemasked(a):
    if False:
        print('Hello World!')
    shape = a.shape
    reps = shape[-1] // 3
    msk = np.empty(shape, dtype=bool)
    msk[...] = [True, True, False] * reps
    it = np.nditer([a, msk], [], [['readwrite', 'writemasked'], ['readonly', 'arraymask']])
    with it:
        for (x, m) in it:
            x[...] = 1
    assert_equal(a, np.broadcast_to([1, 1, 1] * reps, shape))
    it = np.nditer([a, msk], ['buffered'], [['readwrite', 'writemasked'], ['readonly', 'arraymask']])
    is_buffered = True
    with it:
        for (x, m) in it:
            x[...] = 2.5
            if np.may_share_memory(x, a):
                is_buffered = False
    if not is_buffered:
        assert_equal(a, np.broadcast_to([2.5, 2.5, 2.5] * reps, shape))
    else:
        assert_equal(a, np.broadcast_to([2.5, 2.5, 1] * reps, shape))
        a[...] = 2.5
    it = np.nditer([a, msk], ['buffered'], [['readwrite', 'writemasked'], ['readonly', 'arraymask']], op_dtypes=['i8', None], casting='unsafe')
    with it:
        for (x, m) in it:
            x[...] = 3
    assert_equal(a, np.broadcast_to([3, 3, 2.5] * reps, shape))

@pytest.mark.parametrize(['mask', 'mask_axes'], [(None, [-1, 0]), (np.zeros((1, 4), dtype='bool'), [0, 1]), (np.zeros((1, 4), dtype='bool'), None), (np.zeros(4, dtype='bool'), [-1, 0]), (np.zeros((), dtype='bool'), [-1, -1]), (np.zeros((), dtype='bool'), None)])
def test_iter_writemasked_broadcast_error(mask, mask_axes):
    if False:
        print('Hello World!')
    arr = np.zeros((3, 4))
    itflags = ['reduce_ok']
    mask_flags = ['arraymask', 'readwrite', 'allocate']
    a_flags = ['writeonly', 'writemasked']
    if mask_axes is None:
        op_axes = None
    else:
        op_axes = [mask_axes, [0, 1]]
    with assert_raises(ValueError):
        np.nditer((mask, arr), flags=itflags, op_flags=[mask_flags, a_flags], op_axes=op_axes)

def test_iter_writemasked_decref():
    if False:
        for i in range(10):
            print('nop')
    arr = np.arange(10000).astype('>i,O')
    original = arr.copy()
    mask = np.random.randint(0, 2, size=10000).astype(bool)
    it = np.nditer([arr, mask], ['buffered', 'refs_ok'], [['readwrite', 'writemasked'], ['readonly', 'arraymask']], op_dtypes=['<i,O', '?'])
    singleton = object()
    if HAS_REFCOUNT:
        count = sys.getrefcount(singleton)
    for (buf, mask_buf) in it:
        buf[...] = (3, singleton)
    del buf, mask_buf, it
    if HAS_REFCOUNT:
        assert sys.getrefcount(singleton) - count == np.count_nonzero(mask)
    assert_array_equal(arr[~mask], original[~mask])
    assert (arr[mask] == np.array((3, singleton), arr.dtype)).all()
    del arr
    if HAS_REFCOUNT:
        assert sys.getrefcount(singleton) == count

def test_iter_non_writable_attribute_deletion():
    if False:
        print('Hello World!')
    it = np.nditer(np.ones(2))
    attr = ['value', 'shape', 'operands', 'itviews', 'has_delayed_bufalloc', 'iterationneedsapi', 'has_multi_index', 'has_index', 'dtypes', 'ndim', 'nop', 'itersize', 'finished']
    for s in attr:
        assert_raises(AttributeError, delattr, it, s)

def test_iter_writable_attribute_deletion():
    if False:
        for i in range(10):
            print('nop')
    it = np.nditer(np.ones(2))
    attr = ['multi_index', 'index', 'iterrange', 'iterindex']
    for s in attr:
        assert_raises(AttributeError, delattr, it, s)

def test_iter_element_deletion():
    if False:
        return 10
    it = np.nditer(np.ones(3))
    try:
        del it[1]
        del it[1:2]
    except TypeError:
        pass
    except Exception:
        raise AssertionError

def test_iter_allocated_array_dtypes():
    if False:
        for i in range(10):
            print('nop')
    it = np.nditer(([1, 3, 20], None), op_dtypes=[None, ('i4', (2,))])
    for (a, b) in it:
        b[0] = a - 1
        b[1] = a + 1
    assert_equal(it.operands[1], [[0, 2], [2, 4], [19, 21]])
    it = np.nditer(([[1, 3, 20]], None), op_dtypes=[None, ('i4', (2,))], flags=['reduce_ok'], op_axes=[None, (-1, 0)])
    for (a, b) in it:
        b[0] = a - 1
        b[1] = a + 1
    assert_equal(it.operands[1], [[0, 2], [2, 4], [19, 21]])
    it = np.nditer((10, 2, None), op_dtypes=[None, None, ('i4', (2, 2))])
    for (a, b, c) in it:
        c[0, 0] = a - b
        c[0, 1] = a + b
        c[1, 0] = a * b
        c[1, 1] = a / b
    assert_equal(it.operands[2], [[8, 12], [20, 5]])

def test_0d_iter():
    if False:
        i = 10
        return i + 15
    i = nditer([2, 3], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.ndim, 0)
    assert_equal(next(i), (2, 3))
    assert_equal(i.multi_index, ())
    assert_equal(i.iterindex, 0)
    assert_raises(StopIteration, next, i)
    i.reset()
    assert_equal(next(i), (2, 3))
    assert_raises(StopIteration, next, i)
    i = nditer(np.arange(5), ['multi_index'], [['readonly']], op_axes=[()])
    assert_equal(i.ndim, 0)
    assert_equal(len(i), 1)
    i = nditer(np.arange(5), ['multi_index'], [['readonly']], op_axes=[()], itershape=())
    assert_equal(i.ndim, 0)
    assert_equal(len(i), 1)
    with assert_raises(ValueError):
        nditer(np.arange(5), ['multi_index'], [['readonly']], itershape=())
    sdt = [('a', 'f4'), ('b', 'i8'), ('c', 'c8', (2, 3)), ('d', 'O')]
    a = np.array(0.5, dtype='f4')
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt)
    vals = next(i)
    assert_equal(vals['a'], 0.5)
    assert_equal(vals['b'], 0)
    assert_equal(vals['c'], [[0.5] * 3] * 2)
    assert_equal(vals['d'], 0.5)

def test_object_iter_cleanup():
    if False:
        return 10
    assert_raises(TypeError, lambda : np.zeros((17000, 2), dtype='f4') * None)
    arr = np.arange(ncu.BUFSIZE * 10).reshape(10, -1).astype(str)
    oarr = arr.astype(object)
    oarr[:, -1] = None
    assert_raises(TypeError, lambda : np.add(oarr[:, ::-1], arr[:, ::-1]))

    class T:

        def __bool__(self):
            if False:
                i = 10
                return i + 15
            raise TypeError('Ambiguous')
    assert_raises(TypeError, np.logical_or.reduce, np.array([T(), T()], dtype='O'))

def test_object_iter_cleanup_reduce():
    if False:
        i = 10
        return i + 15
    arr = np.array([[None, 1], [-1, -1], [None, 2], [-1, -1]])[::2]
    with pytest.raises(TypeError):
        np.sum(arr)

@pytest.mark.parametrize('arr', [np.ones((8000, 4, 2), dtype=object)[:, ::2, :], np.ones((8000, 4, 2), dtype=object, order='F')[:, ::2, :], np.ones((8000, 4, 2), dtype=object)[:, ::2, :].copy('F')])
def test_object_iter_cleanup_large_reduce(arr):
    if False:
        for i in range(10):
            print('nop')
    out = np.ones(8000, dtype=np.intp)
    res = np.sum(arr, axis=(1, 2), dtype=object, out=out)
    assert_array_equal(res, np.full(8000, 4, dtype=object))

def test_iter_too_large():
    if False:
        print('Hello World!')
    size = np.iinfo(np.intp).max // 1024
    arr = np.lib.stride_tricks.as_strided(np.zeros(1), (size,), (0,))
    assert_raises(ValueError, nditer, (arr, arr[:, None]))
    assert_raises(ValueError, nditer, (arr, arr[:, None]), flags=['multi_index'])

def test_iter_too_large_with_multiindex():
    if False:
        while True:
            i = 10
    base_size = 2 ** 10
    num = 1
    while base_size ** num < np.iinfo(np.intp).max:
        num += 1
    shape_template = [1, 1] * num
    arrays = []
    for i in range(num):
        shape = shape_template[:]
        shape[i * 2] = 2 ** 10
        arrays.append(np.empty(shape))
    arrays = tuple(arrays)
    for mode in range(6):
        with assert_raises(ValueError):
            _multiarray_tests.test_nditer_too_large(arrays, -1, mode)
    _multiarray_tests.test_nditer_too_large(arrays, -1, 7)
    for i in range(num):
        for mode in range(6):
            _multiarray_tests.test_nditer_too_large(arrays, i * 2, mode)
            with assert_raises(ValueError):
                _multiarray_tests.test_nditer_too_large(arrays, i * 2 + 1, mode)

def test_writebacks():
    if False:
        return 10
    a = np.arange(6, dtype='f4')
    au = a.byteswap()
    au = au.view(au.dtype.newbyteorder())
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    it = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    with it:
        it.operands[0][:] = 100
    assert_equal(au, 100)
    it = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    try:
        with it:
            assert_equal(au.flags.writeable, False)
            it.operands[0][:] = 0
            raise ValueError('exit context manager on exception')
    except:
        pass
    assert_equal(au, 0)
    assert_equal(au.flags.writeable, True)
    assert_raises(ValueError, getattr, it, 'operands')
    it = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    with it:
        x = it.operands[0]
        x[:] = 6
        assert_(x.flags.writebackifcopy)
    assert_equal(au, 6)
    assert_(not x.flags.writebackifcopy)
    x[:] = 123
    assert_equal(au, 6)
    it = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    with it:
        with it:
            for x in it:
                x[...] = 123
    it = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    with it:
        with it:
            for x in it:
                x[...] = 123
        assert_raises(ValueError, getattr, it, 'operands')
    it = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    del au
    with it:
        for x in it:
            x[...] = 123
    enter = it.__enter__
    assert_raises(RuntimeError, enter)

def test_close_equivalent():
    if False:
        return 10
    ' using a context amanger and using nditer.close are equivalent\n    '

    def add_close(x, y, out=None):
        if False:
            print('Hello World!')
        addop = np.add
        it = np.nditer([x, y, out], [], [['readonly'], ['readonly'], ['writeonly', 'allocate']])
        for (a, b, c) in it:
            addop(a, b, out=c)
        ret = it.operands[2]
        it.close()
        return ret

    def add_context(x, y, out=None):
        if False:
            for i in range(10):
                print('nop')
        addop = np.add
        it = np.nditer([x, y, out], [], [['readonly'], ['readonly'], ['writeonly', 'allocate']])
        with it:
            for (a, b, c) in it:
                addop(a, b, out=c)
            return it.operands[2]
    z = add_close(range(5), range(5))
    assert_equal(z, range(0, 10, 2))
    z = add_context(range(5), range(5))
    assert_equal(z, range(0, 10, 2))

def test_close_raises():
    if False:
        while True:
            i = 10
    it = np.nditer(np.arange(3))
    assert_equal(next(it), 0)
    it.close()
    assert_raises(StopIteration, next, it)
    assert_raises(ValueError, getattr, it, 'operands')

def test_close_parameters():
    if False:
        i = 10
        return i + 15
    it = np.nditer(np.arange(3))
    assert_raises(TypeError, it.close, 1)

@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_warn_noclose():
    if False:
        print('Hello World!')
    a = np.arange(6, dtype='f4')
    au = a.byteswap()
    au = au.view(au.dtype.newbyteorder())
    with suppress_warnings() as sup:
        sup.record(RuntimeWarning)
        it = np.nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
        del it
        assert len(sup.log) == 1

@pytest.mark.skipif(sys.version_info[:2] == (3, 9) and sys.platform == 'win32', reason='Errors with Python 3.9 on Windows')
@pytest.mark.parametrize(['in_dtype', 'buf_dtype'], [('i', 'O'), ('O', 'i'), ('i,O', 'O,O'), ('O,i', 'i,O')])
@pytest.mark.parametrize('steps', [1, 2, 3])
def test_partial_iteration_cleanup(in_dtype, buf_dtype, steps):
    if False:
        while True:
            i = 10
    '\n    Checks for reference counting leaks during cleanup.  Using explicit\n    reference counts lead to occasional false positives (at least in parallel\n    test setups).  This test now should still test leaks correctly when\n    run e.g. with pytest-valgrind or pytest-leaks\n    '
    value = 2 ** 30 + 1
    arr = np.full(int(ncu.BUFSIZE * 2.5), value).astype(in_dtype)
    it = np.nditer(arr, op_dtypes=[np.dtype(buf_dtype)], flags=['buffered', 'external_loop', 'refs_ok'], casting='unsafe')
    for step in range(steps):
        next(it)
    del it
    it = np.nditer(arr, op_dtypes=[np.dtype(buf_dtype)], flags=['buffered', 'external_loop', 'refs_ok'], casting='unsafe')
    for step in range(steps):
        it.iternext()
    del it

@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
@pytest.mark.parametrize(['in_dtype', 'buf_dtype'], [('O', 'i'), ('O,i', 'i,O')])
def test_partial_iteration_error(in_dtype, buf_dtype):
    if False:
        for i in range(10):
            print('nop')
    value = 123
    arr = np.full(int(ncu.BUFSIZE * 2.5), value).astype(in_dtype)
    if in_dtype == 'O':
        arr[int(ncu.BUFSIZE * 1.5)] = None
    else:
        arr[int(ncu.BUFSIZE * 1.5)]['f0'] = None
    count = sys.getrefcount(value)
    it = np.nditer(arr, op_dtypes=[np.dtype(buf_dtype)], flags=['buffered', 'external_loop', 'refs_ok'], casting='unsafe')
    with pytest.raises(TypeError):
        next(it)
        next(it)
    it.reset()
    with pytest.raises(TypeError):
        it.iternext()
        it.iternext()
    assert count == sys.getrefcount(value)

def test_debug_print(capfd):
    if False:
        for i in range(10):
            print('nop')
    '\n    Matches the expected output of a debug print with the actual output.\n    Note that the iterator dump should not be considered stable API,\n    this test is mainly to ensure the print does not crash.\n\n    Currently uses a subprocess to avoid dealing with the C level `printf`s.\n    '
    expected = "\n    ------ BEGIN ITERATOR DUMP ------\n    | Iterator Address:\n    | ItFlags: BUFFER REDUCE REUSE_REDUCE_LOOPS\n    | NDim: 2\n    | NOp: 2\n    | IterSize: 50\n    | IterStart: 0\n    | IterEnd: 50\n    | IterIndex: 0\n    | Iterator SizeOf:\n    | BufferData SizeOf:\n    | AxisData SizeOf:\n    |\n    | Perm: 0 1\n    | DTypes:\n    | DTypes: dtype('float64') dtype('int32')\n    | InitDataPtrs:\n    | BaseOffsets: 0 0\n    | Operands:\n    | Operand DTypes: dtype('int64') dtype('float64')\n    | OpItFlags:\n    |   Flags[0]: READ CAST ALIGNED\n    |   Flags[1]: READ WRITE CAST ALIGNED REDUCE\n    |\n    | BufferData:\n    |   BufferSize: 50\n    |   Size: 5\n    |   BufIterEnd: 5\n    |   REDUCE Pos: 0\n    |   REDUCE OuterSize: 10\n    |   REDUCE OuterDim: 1\n    |   Strides: 8 4\n    |   Ptrs:\n    |   REDUCE Outer Strides: 40 0\n    |   REDUCE Outer Ptrs:\n    |   ReadTransferFn:\n    |   ReadTransferData:\n    |   WriteTransferFn:\n    |   WriteTransferData:\n    |   Buffers:\n    |\n    | AxisData[0]:\n    |   Shape: 5\n    |   Index: 0\n    |   Strides: 16 8\n    |   Ptrs:\n    | AxisData[1]:\n    |   Shape: 10\n    |   Index: 0\n    |   Strides: 80 0\n    |   Ptrs:\n    ------- END ITERATOR DUMP -------\n    ".strip().splitlines()
    arr1 = np.arange(100, dtype=np.int64).reshape(10, 10)[:, ::2]
    arr2 = np.arange(5.0)
    it = np.nditer((arr1, arr2), op_dtypes=['d', 'i4'], casting='unsafe', flags=['reduce_ok', 'buffered'], op_flags=[['readonly'], ['readwrite']])
    it.debug_print()
    res = capfd.readouterr().out
    res = res.strip().splitlines()
    assert len(res) == len(expected)
    for (res_line, expected_line) in zip(res, expected):
        assert res_line.startswith(expected_line.strip())