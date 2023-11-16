import functools
import sys
from unittest import expectedFailure as xfail, skipIf as skipif
from pytest import raises as assert_raises
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TEST_WITH_TORCHDYNAMO, TestCase, xfailIfTorchDynamo, xpassIfTorchDynamo
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import apply_along_axis, array_split, column_stack, dsplit, dstack, expand_dims, hsplit, kron, put_along_axis, split, take_along_axis, tile, vsplit
    from numpy.random import rand, randint
    from numpy.testing import assert_, assert_array_equal, assert_equal
else:
    import torch._numpy as np
    from torch._numpy import array_split, column_stack, dsplit, dstack, expand_dims, hsplit, kron, put_along_axis, split, take_along_axis, tile, vsplit
    from torch._numpy.random import rand, randint
    from torch._numpy.testing import assert_, assert_array_equal, assert_equal
skip = functools.partial(skipif, True)
IS_64BIT = sys.maxsize > 2 ** 32

def _add_keepdims(func):
    if False:
        i = 10
        return i + 15
    'hack in keepdims behavior into a function taking an axis'

    @functools.wraps(func)
    def wrapped(a, axis, **kwargs):
        if False:
            print('Hello World!')
        res = func(a, axis=axis, **kwargs)
        if axis is None:
            axis = 0
        return np.expand_dims(res, axis=axis)
    return wrapped

class TestTakeAlongAxis(TestCase):

    def test_argequivalent(self):
        if False:
            return 10
        'Test it translates from arg<func> to <func>'
        a = rand(3, 4, 5)
        funcs = [(np.sort, np.argsort, dict()), (_add_keepdims(np.min), _add_keepdims(np.argmin), dict()), (_add_keepdims(np.max), _add_keepdims(np.argmax), dict())]
        for (func, argfunc, kwargs) in funcs:
            for axis in list(range(a.ndim)) + [None]:
                a_func = func(a, axis=axis, **kwargs)
                ai_func = argfunc(a, axis=axis, **kwargs)
                assert_equal(a_func, take_along_axis(a, ai_func, axis=axis))

    def test_invalid(self):
        if False:
            print('Hello World!')
        'Test it errors when indices has too few dimensions'
        a = np.ones((10, 10))
        ai = np.ones((10, 2), dtype=np.intp)
        take_along_axis(a, ai, axis=1)
        assert_raises((ValueError, RuntimeError), take_along_axis, a, np.array(1), axis=1)
        assert_raises((IndexError, RuntimeError), take_along_axis, a, ai.astype(bool), axis=1)
        assert_raises((IndexError, RuntimeError), take_along_axis, a, ai.astype(float), axis=1)
        assert_raises(np.AxisError, take_along_axis, a, ai, axis=10)

    def test_empty(self):
        if False:
            print('Hello World!')
        'Test everything is ok with empty results, even with inserted dims'
        a = np.ones((3, 4, 5))
        ai = np.ones((3, 0, 5), dtype=np.intp)
        actual = take_along_axis(a, ai, axis=1)
        assert_equal(actual.shape, ai.shape)

    def test_broadcast(self):
        if False:
            print('Hello World!')
        'Test that non-indexing dimensions are broadcast in both directions'
        a = np.ones((3, 4, 1))
        ai = np.ones((1, 2, 5), dtype=np.intp)
        actual = take_along_axis(a, ai, axis=1)
        assert_equal(actual.shape, (3, 2, 5))

class TestPutAlongAxis(TestCase):

    def test_replace_max(self):
        if False:
            i = 10
            return i + 15
        a_base = np.array([[10, 30, 20], [60, 40, 50]])
        for axis in list(range(a_base.ndim)) + [None]:
            a = a_base.copy()
            i_max = _add_keepdims(np.argmax)(a, axis=axis)
            put_along_axis(a, i_max, -99, axis=axis)
            i_min = _add_keepdims(np.argmin)(a, axis=axis)
            assert_equal(i_min, i_max)

    @xpassIfTorchDynamo
    def test_broadcast(self):
        if False:
            i = 10
            return i + 15
        'Test that non-indexing dimensions are broadcast in both directions'
        a = np.ones((3, 4, 1))
        ai = np.arange(10, dtype=np.intp).reshape((1, 2, 5)) % 4
        put_along_axis(a, ai, 20, axis=1)
        assert_equal(take_along_axis(a, ai, axis=1), 20)

@xpassIfTorchDynamo
class TestApplyAlongAxis(TestCase):

    def test_simple(self):
        if False:
            print('Hello World!')
        a = np.ones((20, 10), 'd')
        assert_array_equal(apply_along_axis(len, 0, a), len(a) * np.ones(a.shape[1]))

    def test_simple101(self):
        if False:
            return 10
        a = np.ones((10, 101), 'd')
        assert_array_equal(apply_along_axis(len, 0, a), len(a) * np.ones(a.shape[1]))

    def test_3d(self):
        if False:
            print('Hello World!')
        a = np.arange(27).reshape((3, 3, 3))
        assert_array_equal(apply_along_axis(np.sum, 0, a), [[27, 30, 33], [36, 39, 42], [45, 48, 51]])

    def test_scalar_array(self, cls=np.ndarray):
        if False:
            i = 10
            return i + 15
        a = np.ones((6, 3)).view(cls)
        res = apply_along_axis(np.sum, 0, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))

    def test_0d_array(self, cls=np.ndarray):
        if False:
            return 10

        def sum_to_0d(x):
            if False:
                for i in range(10):
                    print('nop')
            'Sum x, returning a 0d array of the same class'
            assert_equal(x.ndim, 1)
            return np.squeeze(np.sum(x, keepdims=True))
        a = np.ones((6, 3)).view(cls)
        res = apply_along_axis(sum_to_0d, 0, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))
        res = apply_along_axis(sum_to_0d, 1, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([3, 3, 3, 3, 3, 3]).view(cls))

    def test_axis_insertion(self, cls=np.ndarray):
        if False:
            return 10

        def f1to2(x):
            if False:
                i = 10
                return i + 15
            'produces an asymmetric non-square matrix from x'
            assert_equal(x.ndim, 1)
            return (x[::-1] * x[1:, None]).view(cls)
        a2d = np.arange(6 * 3).reshape((6, 3))
        actual = apply_along_axis(f1to2, 0, a2d)
        expected = np.stack([f1to2(a2d[:, i]) for i in range(a2d.shape[1])], axis=-1).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)
        actual = apply_along_axis(f1to2, 1, a2d)
        expected = np.stack([f1to2(a2d[i, :]) for i in range(a2d.shape[0])], axis=0).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)
        a3d = np.arange(6 * 5 * 3).reshape((6, 5, 3))
        actual = apply_along_axis(f1to2, 1, a3d)
        expected = np.stack([np.stack([f1to2(a3d[i, :, j]) for i in range(a3d.shape[0])], axis=0) for j in range(a3d.shape[2])], axis=-1).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)

    def test_axis_insertion_ma(self):
        if False:
            while True:
                i = 10

        def f1to2(x):
            if False:
                print('Hello World!')
            'produces an asymmetric non-square matrix from x'
            assert_equal(x.ndim, 1)
            res = x[::-1] * x[1:, None]
            return np.ma.masked_where(res % 5 == 0, res)
        a = np.arange(6 * 3).reshape((6, 3))
        res = apply_along_axis(f1to2, 0, a)
        assert_(isinstance(res, np.ma.masked_array))
        assert_equal(res.ndim, 3)
        assert_array_equal(res[:, :, 0].mask, f1to2(a[:, 0]).mask)
        assert_array_equal(res[:, :, 1].mask, f1to2(a[:, 1]).mask)
        assert_array_equal(res[:, :, 2].mask, f1to2(a[:, 2]).mask)

    def test_tuple_func1d(self):
        if False:
            print('Hello World!')

        def sample_1d(x):
            if False:
                for i in range(10):
                    print('nop')
            return (x[1], x[0])
        res = np.apply_along_axis(sample_1d, 1, np.array([[1, 2], [3, 4]]))
        assert_array_equal(res, np.array([[2, 1], [4, 3]]))

    def test_empty(self):
        if False:
            print('Hello World!')

        def never_call(x):
            if False:
                while True:
                    i = 10
            assert_(False)
        a = np.empty((0, 0))
        assert_raises(ValueError, np.apply_along_axis, never_call, 0, a)
        assert_raises(ValueError, np.apply_along_axis, never_call, 1, a)

        def empty_to_1(x):
            if False:
                print('Hello World!')
            assert_(len(x) == 0)
            return 1
        a = np.empty((10, 0))
        actual = np.apply_along_axis(empty_to_1, 1, a)
        assert_equal(actual, np.ones(10))
        assert_raises(ValueError, np.apply_along_axis, empty_to_1, 0, a)

    def test_with_iterable_object(self):
        if False:
            while True:
                i = 10
        d = np.array([[{1, 11}, {2, 22}, {3, 33}], [{4, 44}, {5, 55}, {6, 66}]])
        actual = np.apply_along_axis(lambda a: set.union(*a), 0, d)
        expected = np.array([{1, 11, 4, 44}, {2, 22, 5, 55}, {3, 33, 6, 66}])
        assert_equal(actual, expected)
        for i in np.ndindex(actual.shape):
            assert_equal(type(actual[i]), type(expected[i]))

@xfail
class TestApplyOverAxes(TestCase):

    def test_simple(self):
        if False:
            while True:
                i = 10
        a = np.arange(24).reshape(2, 3, 4)
        aoa_a = apply_over_axes(np.sum, a, [0, 2])
        assert_array_equal(aoa_a, np.array([[[60], [92], [124]]]))

class TestExpandDims(TestCase):

    def test_functionality(self):
        if False:
            i = 10
            return i + 15
        s = (2, 3, 4, 5)
        a = np.empty(s)
        for axis in range(-5, 4):
            b = expand_dims(a, axis)
            assert_(b.shape[axis] == 1)
            assert_(np.squeeze(b).shape == s)

    def test_axis_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.empty((3, 3, 3))
        assert np.expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
        assert np.expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
        assert np.expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
        assert np.expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)

    def test_axis_out_of_range(self):
        if False:
            i = 10
            return i + 15
        s = (2, 3, 4, 5)
        a = np.empty(s)
        assert_raises(np.AxisError, expand_dims, a, -6)
        assert_raises(np.AxisError, expand_dims, a, 5)
        a = np.empty((3, 3, 3))
        assert_raises(np.AxisError, expand_dims, a, (0, -6))
        assert_raises(np.AxisError, expand_dims, a, (0, 5))

    def test_repeated_axis(self):
        if False:
            while True:
                i = 10
        a = np.empty((3, 3, 3))
        assert_raises(ValueError, expand_dims, a, axis=(1, 1))

class TestArraySplit(TestCase):

    def test_integer_0_split(self):
        if False:
            i = 10
            return i + 15
        a = np.arange(10)
        assert_raises(ValueError, array_split, a, 0)

    def test_integer_split(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.arange(10)
        res = array_split(a, 1)
        desired = [np.arange(10)]
        compare_results(res, desired)
        res = array_split(a, 2)
        desired = [np.arange(5), np.arange(5, 10)]
        compare_results(res, desired)
        res = array_split(a, 3)
        desired = [np.arange(4), np.arange(4, 7), np.arange(7, 10)]
        compare_results(res, desired)
        res = array_split(a, 4)
        desired = [np.arange(3), np.arange(3, 6), np.arange(6, 8), np.arange(8, 10)]
        compare_results(res, desired)
        res = array_split(a, 5)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6), np.arange(6, 8), np.arange(8, 10)]
        compare_results(res, desired)
        res = array_split(a, 6)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6), np.arange(6, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)
        res = array_split(a, 7)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)
        res = array_split(a, 8)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 5), np.arange(5, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)
        res = array_split(a, 9)
        desired = [np.arange(2), np.arange(2, 3), np.arange(3, 4), np.arange(4, 5), np.arange(5, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)
        res = array_split(a, 10)
        desired = [np.arange(1), np.arange(1, 2), np.arange(2, 3), np.arange(3, 4), np.arange(4, 5), np.arange(5, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)
        res = array_split(a, 11)
        desired = [np.arange(1), np.arange(1, 2), np.arange(2, 3), np.arange(3, 4), np.arange(4, 5), np.arange(5, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10), np.array([])]
        compare_results(res, desired)

    def test_integer_split_2D_rows(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3, axis=0)
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]), np.zeros((0, 10))]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)
        res = array_split(a, [0, 1], axis=0)
        tgt = [np.zeros((0, 10)), np.array([np.arange(10)]), np.array([np.arange(10)])]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

    def test_integer_split_2D_cols(self):
        if False:
            i = 10
            return i + 15
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3, axis=-1)
        desired = [np.array([np.arange(4), np.arange(4)]), np.array([np.arange(4, 7), np.arange(4, 7)]), np.array([np.arange(7, 10), np.arange(7, 10)])]
        compare_results(res, desired)

    def test_integer_split_2D_default(self):
        if False:
            while True:
                i = 10
        'This will fail if we change default axis'
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3)
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]), np.zeros((0, 10))]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

    @skipif(not IS_64BIT, reason='Needs 64bit platform')
    def test_integer_split_2D_rows_greater_max_int32(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.broadcast_to([0], (1 << 32, 2))
        res = array_split(a, 4)
        chunk = np.broadcast_to([0], (1 << 30, 2))
        tgt = [chunk] * 4
        for i in range(len(tgt)):
            assert_equal(res[i].shape, tgt[i].shape)

    def test_index_split_simple(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.arange(10)
        indices = [1, 5, 7]
        res = array_split(a, indices, axis=-1)
        desired = [np.arange(0, 1), np.arange(1, 5), np.arange(5, 7), np.arange(7, 10)]
        compare_results(res, desired)

    def test_index_split_low_bound(self):
        if False:
            i = 10
            return i + 15
        a = np.arange(10)
        indices = [0, 5, 7]
        res = array_split(a, indices, axis=-1)
        desired = [np.array([]), np.arange(0, 5), np.arange(5, 7), np.arange(7, 10)]
        compare_results(res, desired)

    def test_index_split_high_bound(self):
        if False:
            print('Hello World!')
        a = np.arange(10)
        indices = [0, 5, 7, 10, 12]
        res = array_split(a, indices, axis=-1)
        desired = [np.array([]), np.arange(0, 5), np.arange(5, 7), np.arange(7, 10), np.array([]), np.array([])]
        compare_results(res, desired)

class TestSplit(TestCase):

    def test_equal_split(self):
        if False:
            return 10
        a = np.arange(10)
        res = split(a, 2)
        desired = [np.arange(5), np.arange(5, 10)]
        compare_results(res, desired)

    def test_unequal_split(self):
        if False:
            while True:
                i = 10
        a = np.arange(10)
        assert_raises(ValueError, split, a, 3)

class TestColumnStack(TestCase):

    def test_non_iterable(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(TypeError, column_stack, 1)

    def test_1D_arrays(self):
        if False:
            print('Hello World!')
        a = np.array((1, 2, 3))
        b = np.array((2, 3, 4))
        expected = np.array([[1, 2], [2, 3], [3, 4]])
        actual = np.column_stack((a, b))
        assert_equal(actual, expected)

    def test_2D_arrays(self):
        if False:
            return 10
        a = np.array([[1], [2], [3]])
        b = np.array([[2], [3], [4]])
        expected = np.array([[1, 2], [2, 3], [3, 4]])
        actual = np.column_stack((a, b))
        assert_equal(actual, expected)

    def test_generator(self):
        if False:
            while True:
                i = 10
        column_stack((np.arange(3) for _ in range(2)))

class TestDstack(TestCase):

    def test_non_iterable(self):
        if False:
            return 10
        assert_raises(TypeError, dstack, 1)

    def test_0D_array(self):
        if False:
            return 10
        a = np.array(1)
        b = np.array(2)
        res = dstack([a, b])
        desired = np.array([[[1, 2]]])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        if False:
            i = 10
            return i + 15
        a = np.array([1])
        b = np.array([2])
        res = dstack([a, b])
        desired = np.array([[[1, 2]]])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        if False:
            return 10
        a = np.array([[1], [2]])
        b = np.array([[1], [2]])
        res = dstack([a, b])
        desired = np.array([[[1, 1]], [[2, 2]]])
        assert_array_equal(res, desired)

    def test_2D_array2(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.array([1, 2])
        b = np.array([1, 2])
        res = dstack([a, b])
        desired = np.array([[[1, 1], [2, 2]]])
        assert_array_equal(res, desired)

    def test_generator(self):
        if False:
            return 10
        dstack((np.arange(3) for _ in range(2)))

class TestHsplit(TestCase):
    """Only testing for integer splits."""

    def test_non_iterable(self):
        if False:
            return 10
        assert_raises(ValueError, hsplit, 1, 1)

    def test_0D_array(self):
        if False:
            i = 10
            return i + 15
        a = np.array(1)
        try:
            hsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_1D_array(self):
        if False:
            return 10
        a = np.array([1, 2, 3, 4])
        res = hsplit(a, 2)
        desired = [np.array([1, 2]), np.array([3, 4])]
        compare_results(res, desired)

    def test_2D_array(self):
        if False:
            return 10
        a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        res = hsplit(a, 2)
        desired = [np.array([[1, 2], [1, 2]]), np.array([[3, 4], [3, 4]])]
        compare_results(res, desired)

class TestVsplit(TestCase):
    """Only testing for integer splits."""

    def test_non_iterable(self):
        if False:
            return 10
        assert_raises(ValueError, vsplit, 1, 1)

    def test_0D_array(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.array(1)
        assert_raises(ValueError, vsplit, a, 2)

    def test_1D_array(self):
        if False:
            i = 10
            return i + 15
        a = np.array([1, 2, 3, 4])
        try:
            vsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_2D_array(self):
        if False:
            print('Hello World!')
        a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        res = vsplit(a, 2)
        desired = [np.array([[1, 2, 3, 4]]), np.array([[1, 2, 3, 4]])]
        compare_results(res, desired)

class TestDsplit(TestCase):

    def test_non_iterable(self):
        if False:
            i = 10
            return i + 15
        assert_raises(ValueError, dsplit, 1, 1)

    def test_0D_array(self):
        if False:
            while True:
                i = 10
        a = np.array(1)
        assert_raises(ValueError, dsplit, a, 2)

    def test_1D_array(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.array([1, 2, 3, 4])
        assert_raises(ValueError, dsplit, a, 2)

    def test_2D_array(self):
        if False:
            return 10
        a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        try:
            dsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_3D_array(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
        res = dsplit(a, 2)
        desired = [np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]]), np.array([[[3, 4], [3, 4]], [[3, 4], [3, 4]]])]
        compare_results(res, desired)

class TestSqueeze(TestCase):

    def test_basic(self):
        if False:
            while True:
                i = 10
        a = rand(20, 10, 10, 1, 1)
        b = rand(20, 1, 10, 1, 20)
        c = rand(1, 1, 20, 10)
        assert_array_equal(np.squeeze(a), np.reshape(a, (20, 10, 10)))
        assert_array_equal(np.squeeze(b), np.reshape(b, (20, 10, 20)))
        assert_array_equal(np.squeeze(c), np.reshape(c, (20, 10)))
        a = [[[1.5]]]
        res = np.squeeze(a)
        assert_equal(res, 1.5)
        assert_equal(res.ndim, 0)
        assert type(res) is np.ndarray

    @xfailIfTorchDynamo
    def test_basic_2(self):
        if False:
            for i in range(10):
                print('nop')
        aa = np.ones((3, 1, 4, 1, 1))
        assert aa.squeeze().tensor._base is aa.tensor

    def test_squeeze_axis(self):
        if False:
            for i in range(10):
                print('nop')
        A = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        assert_equal(np.squeeze(A).shape, (3, 3))
        assert_equal(np.squeeze(A, axis=()), A)
        assert_equal(np.squeeze(np.zeros((1, 3, 1))).shape, (3,))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=0).shape, (3, 1))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=-1).shape, (1, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))]).shape, (3,))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=0).shape, (3, 1))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=-1).shape, (1, 3))

    def test_squeeze_type(self):
        if False:
            i = 10
            return i + 15
        a = np.array([3])
        b = np.array(3)
        assert type(a.squeeze()) is np.ndarray
        assert type(b.squeeze()) is np.ndarray

    @skip(reason="XXX: order='F' not implemented")
    def test_squeeze_contiguous(self):
        if False:
            return 10
        a = np.zeros((1, 2)).squeeze()
        b = np.zeros((2, 2, 2), order='F')[:, :, ::2].squeeze()
        assert_(a.flags.c_contiguous)
        assert_(a.flags.f_contiguous)
        assert_(b.flags.f_contiguous)

    @xpassIfTorchDynamo
    def test_squeeze_axis_handling(self):
        if False:
            i = 10
            return i + 15
        with assert_raises(ValueError):
            np.squeeze(np.array([[1], [2], [3]]), axis=0)

@instantiate_parametrized_tests
class TestKron(TestCase):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        a = np.array(1)
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[1, 2], [3, 4]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array(1)
        assert_array_equal(np.kron(a, b), k)
        a = np.array([3])
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[3, 6], [9, 12]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([3])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[[1]], [[2]]])
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[[1]], [[2]]])
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_array_equal(np.kron(a, b), k)

    @skip(reason='NP_VER: fails on CI')
    @parametrize('shape_a,shape_b', [((1, 1), (1, 1)), ((1, 2, 3), (4, 5, 6)), ((2, 2), (2, 2, 2)), ((1, 0), (1, 1)), ((2, 0, 2), (2, 2)), ((2, 0, 0, 2), (2, 0, 2))])
    def test_kron_shape(self, shape_a, shape_b):
        if False:
            i = 10
            return i + 15
        a = np.ones(shape_a)
        b = np.ones(shape_b)
        normalised_shape_a = (1,) * max(0, len(shape_b) - len(shape_a)) + shape_a
        normalised_shape_b = (1,) * max(0, len(shape_a) - len(shape_b)) + shape_b
        expected_shape = np.multiply(normalised_shape_a, normalised_shape_b)
        k = np.kron(a, b)
        assert np.array_equal(k.shape, expected_shape), 'Unexpected shape from kron'

class TestTile(TestCase):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        a = np.array([0, 1, 2])
        b = [[1, 2], [3, 4]]
        assert_equal(tile(a, 2), [0, 1, 2, 0, 1, 2])
        assert_equal(tile(a, (2, 2)), [[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
        assert_equal(tile(a, (1, 2)), [[0, 1, 2, 0, 1, 2]])
        assert_equal(tile(b, 2), [[1, 2, 1, 2], [3, 4, 3, 4]])
        assert_equal(tile(b, (2, 1)), [[1, 2], [3, 4], [1, 2], [3, 4]])
        assert_equal(tile(b, (2, 2)), [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])

    def test_tile_one_repetition_on_array_gh4679(self):
        if False:
            i = 10
            return i + 15
        a = np.arange(5)
        b = tile(a, 1)
        b += 2
        assert_equal(a, np.arange(5))

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.array([[[]]])
        b = np.array([[], []])
        c = tile(b, 2).shape
        d = tile(a, (3, 2, 5)).shape
        assert_equal(c, (2, 0))
        assert_equal(d, (3, 2, 0))

    def test_kroncompare(self):
        if False:
            i = 10
            return i + 15
        reps = [(2,), (1, 2), (2, 1), (2, 2), (2, 3, 2), (3, 2)]
        shape = [(3,), (2, 3), (3, 4, 3), (3, 2, 3), (4, 3, 2, 4), (2, 2)]
        for s in shape:
            b = randint(0, 10, size=s)
            for r in reps:
                a = np.ones(r, b.dtype)
                large = tile(b, r)
                klarge = kron(a, b)
                assert_equal(large, klarge)

@xpassIfTorchDynamo
class TestMayShareMemory(TestCase):

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        d = np.ones((50, 60))
        d2 = np.ones((30, 60, 6))
        assert_(np.may_share_memory(d, d))
        assert_(np.may_share_memory(d, d[::-1]))
        assert_(np.may_share_memory(d, d[::2]))
        assert_(np.may_share_memory(d, d[1:, ::-1]))
        assert_(not np.may_share_memory(d[::-1], d2))
        assert_(not np.may_share_memory(d[::2], d2))
        assert_(not np.may_share_memory(d[1:, ::-1], d2))
        assert_(np.may_share_memory(d2[1:, ::-1], d2))

def compare_results(res, desired):
    if False:
        return 10
    'Compare lists of arrays.'
    if len(res) != len(desired):
        raise ValueError('Iterables have different lengths')
    for (x, y) in zip(res, desired):
        assert_array_equal(x, y)
if __name__ == '__main__':
    run_tests()