import functools
from unittest import expectedFailure as xfail, skipIf as skipif
import numpy
import pytest
from pytest import raises as assert_raises
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TEST_WITH_TORCHDYNAMO, TestCase, xpassIfTorchDynamo
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import array, atleast_1d, atleast_2d, atleast_3d, AxisError, concatenate, hstack, newaxis, stack, vstack
    from numpy.testing import assert_, assert_array_equal, assert_equal
else:
    import torch._numpy as np
    from torch._numpy import array, atleast_1d, atleast_2d, atleast_3d, AxisError, concatenate, hstack, newaxis, stack, vstack
    from torch._numpy.testing import assert_, assert_array_equal, assert_equal
skip = functools.partial(skipif, True)
IS_PYPY = False

class TestAtleast1d(TestCase):

    def test_0D_array(self):
        if False:
            for i in range(10):
                print('nop')
        a = array(1)
        b = array(2)
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [array([1]), array([2])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        if False:
            i = 10
            return i + 15
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [array([1, 2]), array([2, 3])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        if False:
            i = 10
            return i + 15
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        if False:
            i = 10
            return i + 15
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_r1array(self):
        if False:
            while True:
                i = 10
        "Test to make sure equivalent Travis O's r1array function"
        assert atleast_1d(3).shape == (1,)
        assert atleast_1d(3j).shape == (1,)
        assert atleast_1d(3.0).shape == (1,)
        assert atleast_1d([[2, 3], [4, 5]]).shape == (2, 2)

class TestAtleast2d(TestCase):

    def test_0D_array(self):
        if False:
            while True:
                i = 10
        a = array(1)
        b = array(2)
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [array([[1]]), array([[2]])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        if False:
            for i in range(10):
                print('nop')
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [array([[1, 2]]), array([[2, 3]])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        if False:
            for i in range(10):
                print('nop')
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        if False:
            i = 10
            return i + 15
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_r2array(self):
        if False:
            i = 10
            return i + 15
        "Test to make sure equivalent Travis O's r2array function"
        assert atleast_2d(3).shape == (1, 1)
        assert atleast_2d([3j, 1]).shape == (1, 2)
        assert atleast_2d([[[3, 1], [4, 5]], [[3, 5], [1, 2]]]).shape == (2, 2, 2)

class TestAtleast3d(TestCase):

    def test_0D_array(self):
        if False:
            print('Hello World!')
        a = array(1)
        b = array(2)
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [array([[[1]]]), array([[[2]]])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        if False:
            i = 10
            return i + 15
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [array([[[1], [2]]]), array([[[2], [3]]])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        if False:
            for i in range(10):
                print('nop')
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [a[:, :, newaxis], b[:, :, newaxis]]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        if False:
            while True:
                i = 10
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

class TestHstack(TestCase):

    def test_non_iterable(self):
        if False:
            return 10
        assert_raises(TypeError, hstack, 1)

    def test_empty_input(self):
        if False:
            print('Hello World!')
        assert_raises(ValueError, hstack, ())

    def test_0D_array(self):
        if False:
            print('Hello World!')
        a = array(1)
        b = array(2)
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        if False:
            i = 10
            return i + 15
        a = array([1])
        b = array([2])
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        if False:
            for i in range(10):
                print('nop')
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = hstack([a, b])
        desired = array([[1, 1], [2, 2]])
        assert_array_equal(res, desired)

    def test_generator(self):
        if False:
            while True:
                i = 10
        hstack((np.arange(3) for _ in range(2)))
        hstack((x for x in np.ones((3, 2))))

    @skipif(numpy.__version__ < '1.24', reason='NP_VER: fails on NumPy 1.23.x')
    def test_casting_and_dtype(self):
        if False:
            print('Hello World!')
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.hstack((a, b), casting='unsafe', dtype=np.int64)
        expected_res = np.array([1, 2, 3, 2, 3, 4])
        assert_array_equal(res, expected_res)

    def test_casting_and_dtype_type_error(self):
        if False:
            return 10
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            hstack((a, b), casting='safe', dtype=np.int64)

class TestVstack(TestCase):

    def test_non_iterable(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(TypeError, vstack, 1)

    def test_empty_input(self):
        if False:
            print('Hello World!')
        assert_raises(ValueError, vstack, ())

    def test_0D_array(self):
        if False:
            return 10
        a = array(1)
        b = array(2)
        res = vstack([a, b])
        desired = array([[1], [2]])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        if False:
            print('Hello World!')
        a = array([1])
        b = array([2])
        res = vstack([a, b])
        desired = array([[1], [2]])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        if False:
            i = 10
            return i + 15
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = vstack([a, b])
        desired = array([[1], [2], [1], [2]])
        assert_array_equal(res, desired)

    def test_2D_array2(self):
        if False:
            i = 10
            return i + 15
        a = array([1, 2])
        b = array([1, 2])
        res = vstack([a, b])
        desired = array([[1, 2], [1, 2]])
        assert_array_equal(res, desired)

    @xfail
    def test_generator(self):
        if False:
            print('Hello World!')
        with pytest.raises(TypeError, match='arrays to stack must be'):
            vstack((np.arange(3) for _ in range(2)))

    @skipif(numpy.__version__ < '1.24', reason='casting kwarg is new in NumPy 1.24')
    def test_casting_and_dtype(self):
        if False:
            print('Hello World!')
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.vstack((a, b), casting='unsafe', dtype=np.int64)
        expected_res = np.array([[1, 2, 3], [2, 3, 4]])
        assert_array_equal(res, expected_res)

    @skipif(numpy.__version__ < '1.24', reason='casting kwarg is new in NumPy 1.24')
    def test_casting_and_dtype_type_error(self):
        if False:
            return 10
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            vstack((a, b), casting='safe', dtype=np.int64)

@instantiate_parametrized_tests
class TestConcatenate(TestCase):

    def test_out_and_dtype_simple(self):
        if False:
            for i in range(10):
                print('nop')
        (a, b, out) = (np.ones(3), np.ones(4), np.ones(3 + 4))
        with pytest.raises(TypeError):
            np.concatenate((a, b), out=out, dtype=float)

    def test_returns_copy(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.eye(3)
        b = np.concatenate([a])
        b[0, 0] = 2
        assert b[0, 0] != a[0, 0]

    def test_exceptions(self):
        if False:
            while True:
                i = 10
        for ndim in [1, 2, 3]:
            a = np.ones((1,) * ndim)
            np.concatenate((a, a), axis=0)
            assert_raises((IndexError, np.AxisError), np.concatenate, (a, a), axis=ndim)
            assert_raises((IndexError, np.AxisError), np.concatenate, (a, a), axis=-(ndim + 1))
        assert_raises((RuntimeError, ValueError), concatenate, (0,))
        assert_raises((RuntimeError, ValueError), concatenate, (np.array(0),))
        assert_raises((RuntimeError, ValueError), np.concatenate, (np.zeros(1), np.zeros((1, 1))))
        a = np.ones((1, 2, 3))
        b = np.ones((2, 2, 3))
        axis = list(range(3))
        for i in range(3):
            np.concatenate((a, b), axis=axis[0])
            assert_raises((RuntimeError, ValueError), np.concatenate, (a, b), axis=axis[1])
            assert_raises((RuntimeError, ValueError), np.concatenate, (a, b), axis=axis[2])
            a = np.moveaxis(a, -1, 0)
            b = np.moveaxis(b, -1, 0)
            axis.append(axis.pop(0))
        assert_raises(ValueError, concatenate, ())

    def test_concatenate_axis_None(self):
        if False:
            while True:
                i = 10
        a = np.arange(4, dtype=np.float64).reshape((2, 2))
        b = list(range(3))
        r = np.concatenate((a, a), axis=None)
        assert r.dtype == a.dtype
        assert r.ndim == 1
        r = np.concatenate((a, b), axis=None)
        assert r.size == a.size + len(b)
        assert r.dtype == a.dtype
        out = np.zeros(a.size + len(b))
        r = np.concatenate((a, b), axis=None)
        rout = np.concatenate((a, b), axis=None, out=out)
        assert out is rout
        assert np.all(r == rout)

    @xpassIfTorchDynamo
    def test_large_concatenate_axis_None(self):
        if False:
            i = 10
            return i + 15
        x = np.arange(1, 100)
        r = np.concatenate(x, None)
        assert np.all(x == r)
        r = np.concatenate(x, 100)
        assert_array_equal(x, r)

    def test_concatenate(self):
        if False:
            i = 10
            return i + 15
        r4 = list(range(4))
        r3 = list(range(3))
        assert_array_equal(concatenate((r4, r3)), r4 + r3)
        assert_array_equal(concatenate((tuple(r4), r3)), r4 + r3)
        assert_array_equal(concatenate((array(r4), r3)), r4 + r3)
        assert_array_equal(concatenate((r4, r3), 0), r4 + r3)
        assert_array_equal(concatenate((r4, r3), -1), r4 + r3)
        a23 = array([[10, 11, 12], [13, 14, 15]])
        a13 = array([[0, 1, 2]])
        res = array([[10, 11, 12], [13, 14, 15], [0, 1, 2]])
        assert_array_equal(concatenate((a23, a13)), res)
        assert_array_equal(concatenate((a23, a13), 0), res)
        assert_array_equal(concatenate((a23.T, a13.T), 1), res.T)
        assert_array_equal(concatenate((a23.T, a13.T), -1), res.T)
        assert_raises((RuntimeError, ValueError), concatenate, (a23.T, a13.T), 0)
        res = np.arange(2 * 3 * 7).reshape((2, 3, 7))
        a0 = res[..., :4]
        a1 = res[..., 4:6]
        a2 = res[..., 6:]
        assert_array_equal(concatenate((a0, a1, a2), 2), res)
        assert_array_equal(concatenate((a0, a1, a2), -1), res)
        assert_array_equal(concatenate((a0.T, a1.T, a2.T), 0), res.T)
        out = res.copy()
        rout = concatenate((a0, a1, a2), 2, out=out)
        assert_(out is rout)
        assert_equal(res, rout)

    @skip(reason='concat, arrays, sequence')
    @skipif(IS_PYPY, reason='PYPY handles sq_concat, nb_add differently than cpython')
    def test_operator_concat(self):
        if False:
            return 10
        import operator
        a = array([1, 2])
        b = array([3, 4])
        n = [1, 2]
        res = array([1, 2, 3, 4])
        assert_raises(TypeError, operator.concat, a, b)
        assert_raises(TypeError, operator.concat, a, n)
        assert_raises(TypeError, operator.concat, n, a)
        assert_raises(TypeError, operator.concat, a, 1)
        assert_raises(TypeError, operator.concat, 1, a)

    def test_bad_out_shape(self):
        if False:
            return 10
        a = array([1, 2])
        b = array([3, 4])
        assert_raises(ValueError, concatenate, (a, b), out=np.empty(5))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((4, 1)))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((1, 4)))
        concatenate((a, b), out=np.empty(4))

    @parametrize('axis', [None, 0])
    @parametrize('out_dtype', ['c8', 'f4', 'f8', 'i8'])
    @parametrize('casting', ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
    def test_out_and_dtype(self, axis, out_dtype, casting):
        if False:
            while True:
                i = 10
        out = np.empty(4, dtype=out_dtype)
        to_concat = (array([1.1, 2.2]), array([3.3, 4.4]))
        if not np.can_cast(to_concat[0], out_dtype, casting=casting):
            with assert_raises(TypeError):
                concatenate(to_concat, out=out, axis=axis, casting=casting)
            with assert_raises(TypeError):
                concatenate(to_concat, dtype=out.dtype, axis=axis, casting=casting)
        else:
            res_out = concatenate(to_concat, out=out, axis=axis, casting=casting)
            res_dtype = concatenate(to_concat, dtype=out.dtype, axis=axis, casting=casting)
            assert res_out is out
            assert_array_equal(out, res_dtype)
            assert res_dtype.dtype == out_dtype
        with assert_raises(TypeError):
            concatenate(to_concat, out=out, dtype=out_dtype, axis=axis)

@instantiate_parametrized_tests
class TestStackMisc(TestCase):

    @skipif(numpy.__version__ < '1.24', reason='NP_VER: fails on NumPy 1.23.x')
    def test_stack(self):
        if False:
            print('Hello World!')
        assert_raises(TypeError, stack, 1)
        for input_ in [(1, 2, 3), [np.int32(1), np.int32(2), np.int32(3)], [np.array(1), np.array(2), np.array(3)]]:
            assert_array_equal(stack(input_), [1, 2, 3])
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        r1 = array([[1, 2, 3], [4, 5, 6]])
        assert_array_equal(np.stack((a, b)), r1)
        assert_array_equal(np.stack((a, b), axis=1), r1.T)
        assert_array_equal(np.stack([a, b]), r1)
        assert_array_equal(np.stack(array([a, b])), r1)
        arrays = [np.random.randn(3) for _ in range(10)]
        axes = [0, 1, -1, -2]
        expected_shapes = [(10, 3), (3, 10), (3, 10), (10, 3)]
        for (axis, expected_shape) in zip(axes, expected_shapes):
            assert_equal(np.stack(arrays, axis).shape, expected_shape)
        assert_raises(AxisError, stack, arrays, axis=2)
        assert_raises(AxisError, stack, arrays, axis=-3)
        arrays = [np.random.randn(3, 4) for _ in range(10)]
        axes = [0, 1, 2, -1, -2, -3]
        expected_shapes = [(10, 3, 4), (3, 10, 4), (3, 4, 10), (3, 4, 10), (3, 10, 4), (10, 3, 4)]
        for (axis, expected_shape) in zip(axes, expected_shapes):
            assert_equal(np.stack(arrays, axis).shape, expected_shape)
        assert stack([[], [], []]).shape == (3, 0)
        assert stack([[], [], []], axis=1).shape == (0, 3)
        out = np.zeros_like(r1)
        np.stack((a, b), out=out)
        assert_array_equal(out, r1)
        assert_raises(ValueError, stack, [])
        assert_raises(ValueError, stack, [])
        assert_raises((RuntimeError, ValueError), stack, [1, np.arange(3)])
        assert_raises((RuntimeError, ValueError), stack, [np.arange(3), 1])
        assert_raises((RuntimeError, ValueError), stack, [np.arange(3), 1], axis=1)
        assert_raises((RuntimeError, ValueError), stack, [np.zeros((3, 3)), np.zeros(3)], axis=1)
        assert_raises((RuntimeError, ValueError), stack, [np.arange(2), np.arange(3)])
        result = stack((x for x in range(3)))
        assert_array_equal(result, np.array([0, 1, 2]))
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.stack((a, b), axis=1, casting='unsafe', dtype=np.int64)
        expected_res = np.array([[1, 2], [2, 3], [3, 4]])
        assert_array_equal(res, expected_res)
        with assert_raises(TypeError):
            stack((a, b), dtype=np.int64, axis=1, casting='safe')

    @skipif(numpy.__version__ < '1.24', reason='NP_VER: fails on NumPy 1.23.x')
    @parametrize('axis', [0])
    @parametrize('out_dtype', ['c8', 'f4', 'f8', 'i8'])
    @parametrize('casting', ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
    def test_stack_out_and_dtype(self, axis, out_dtype, casting):
        if False:
            print('Hello World!')
        to_concat = (array([1, 2]), array([3, 4]))
        res = array([[1, 2], [3, 4]])
        out = np.zeros_like(res)
        if not np.can_cast(to_concat[0], out_dtype, casting=casting):
            with assert_raises(TypeError):
                stack(to_concat, dtype=out_dtype, axis=axis, casting=casting)
        else:
            res_out = stack(to_concat, out=out, axis=axis, casting=casting)
            res_dtype = stack(to_concat, dtype=out_dtype, axis=axis, casting=casting)
            assert res_out is out
            assert_array_equal(out, res_dtype)
            assert res_dtype.dtype == out_dtype
        with assert_raises(TypeError):
            stack(to_concat, out=out, dtype=out_dtype, axis=axis)

@xfail
@instantiate_parametrized_tests
class TestBlock(TestCase):

    @pytest.fixture(params=['block', 'force_concatenate', 'force_slicing'])
    def block(self, request):
        if False:
            print('Hello World!')

        def _block_force_concatenate(arrays):
            if False:
                while True:
                    i = 10
            (arrays, list_ndim, result_ndim, _) = _block_setup(arrays)
            return _block_concatenate(arrays, list_ndim, result_ndim)

        def _block_force_slicing(arrays):
            if False:
                return 10
            (arrays, list_ndim, result_ndim, _) = _block_setup(arrays)
            return _block_slicing(arrays, list_ndim, result_ndim)
        if request.param == 'force_concatenate':
            return _block_force_concatenate
        elif request.param == 'force_slicing':
            return _block_force_slicing
        elif request.param == 'block':
            return block
        else:
            raise ValueError('Unknown blocking request. There is a typo in the tests.')

    def test_returns_copy(self, block):
        if False:
            for i in range(10):
                print('nop')
        a = np.eye(3)
        b = block(a)
        b[0, 0] = 2
        assert b[0, 0] != a[0, 0]

    def test_block_total_size_estimate(self, block):
        if False:
            return 10
        (_, _, _, total_size) = _block_setup([1])
        assert total_size == 1
        (_, _, _, total_size) = _block_setup([[1]])
        assert total_size == 1
        (_, _, _, total_size) = _block_setup([[1, 1]])
        assert total_size == 2
        (_, _, _, total_size) = _block_setup([[1], [1]])
        assert total_size == 2
        (_, _, _, total_size) = _block_setup([[1, 2], [3, 4]])
        assert total_size == 4

    def test_block_simple_row_wise(self, block):
        if False:
            while True:
                i = 10
        a_2d = np.ones((2, 2))
        b_2d = 2 * a_2d
        desired = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        result = block([a_2d, b_2d])
        assert_equal(desired, result)

    def test_block_simple_column_wise(self, block):
        if False:
            while True:
                i = 10
        a_2d = np.ones((2, 2))
        b_2d = 2 * a_2d
        expected = np.array([[1, 1], [1, 1], [2, 2], [2, 2]])
        result = block([[a_2d], [b_2d]])
        assert_equal(expected, result)

    def test_block_with_1d_arrays_row_wise(self, block):
        if False:
            while True:
                i = 10
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        expected = np.array([1, 2, 3, 2, 3, 4])
        result = block([a, b])
        assert_equal(expected, result)

    def test_block_with_1d_arrays_multiple_rows(self, block):
        if False:
            for i in range(10):
                print('nop')
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        expected = np.array([[1, 2, 3, 2, 3, 4], [1, 2, 3, 2, 3, 4]])
        result = block([[a, b], [a, b]])
        assert_equal(expected, result)

    def test_block_with_1d_arrays_column_wise(self, block):
        if False:
            print('Hello World!')
        a_1d = np.array([1, 2, 3])
        b_1d = np.array([2, 3, 4])
        expected = np.array([[1, 2, 3], [2, 3, 4]])
        result = block([[a_1d], [b_1d]])
        assert_equal(expected, result)

    def test_block_mixed_1d_and_2d(self, block):
        if False:
            print('Hello World!')
        a_2d = np.ones((2, 2))
        b_1d = np.array([2, 2])
        result = block([[a_2d], [b_1d]])
        expected = np.array([[1, 1], [1, 1], [2, 2]])
        assert_equal(expected, result)

    def test_block_complicated(self, block):
        if False:
            i = 10
            return i + 15
        one_2d = np.array([[1, 1, 1]])
        two_2d = np.array([[2, 2, 2]])
        three_2d = np.array([[3, 3, 3, 3, 3, 3]])
        four_1d = np.array([4, 4, 4, 4, 4, 4])
        five_0d = np.array(5)
        six_1d = np.array([6, 6, 6, 6, 6])
        zero_2d = np.zeros((2, 6))
        expected = np.array([[1, 1, 1, 2, 2, 2], [3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 4, 4], [5, 6, 6, 6, 6, 6], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        result = block([[one_2d, two_2d], [three_2d], [four_1d], [five_0d, six_1d], [zero_2d]])
        assert_equal(result, expected)

    def test_nested(self, block):
        if False:
            return 10
        one = np.array([1, 1, 1])
        two = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        three = np.array([3, 3, 3])
        four = np.array([4, 4, 4])
        five = np.array(5)
        six = np.array([6, 6, 6, 6, 6])
        zero = np.zeros((2, 6))
        result = block([[block([[one], [three], [four]]), two], [five, six], [zero]])
        expected = np.array([[1, 1, 1, 2, 2, 2], [3, 3, 3, 2, 2, 2], [4, 4, 4, 2, 2, 2], [5, 6, 6, 6, 6, 6], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        assert_equal(result, expected)

    def test_3d(self, block):
        if False:
            print('Hello World!')
        a000 = np.ones((2, 2, 2), int) * 1
        a100 = np.ones((3, 2, 2), int) * 2
        a010 = np.ones((2, 3, 2), int) * 3
        a001 = np.ones((2, 2, 3), int) * 4
        a011 = np.ones((2, 3, 3), int) * 5
        a101 = np.ones((3, 2, 3), int) * 6
        a110 = np.ones((3, 3, 2), int) * 7
        a111 = np.ones((3, 3, 3), int) * 8
        result = block([[[a000, a001], [a010, a011]], [[a100, a101], [a110, a111]]])
        expected = array([[[1, 1, 4, 4, 4], [1, 1, 4, 4, 4], [3, 3, 5, 5, 5], [3, 3, 5, 5, 5], [3, 3, 5, 5, 5]], [[1, 1, 4, 4, 4], [1, 1, 4, 4, 4], [3, 3, 5, 5, 5], [3, 3, 5, 5, 5], [3, 3, 5, 5, 5]], [[2, 2, 6, 6, 6], [2, 2, 6, 6, 6], [7, 7, 8, 8, 8], [7, 7, 8, 8, 8], [7, 7, 8, 8, 8]], [[2, 2, 6, 6, 6], [2, 2, 6, 6, 6], [7, 7, 8, 8, 8], [7, 7, 8, 8, 8], [7, 7, 8, 8, 8]], [[2, 2, 6, 6, 6], [2, 2, 6, 6, 6], [7, 7, 8, 8, 8], [7, 7, 8, 8, 8], [7, 7, 8, 8, 8]]])
        assert_array_equal(result, expected)

    def test_block_with_mismatched_shape(self, block):
        if False:
            return 10
        a = np.array([0, 0])
        b = np.eye(2)
        assert_raises(ValueError, block, [a, b])
        assert_raises(ValueError, block, [b, a])
        to_block = [[np.ones((2, 3)), np.ones((2, 2))], [np.ones((2, 2)), np.ones((2, 2))]]
        assert_raises(ValueError, block, to_block)

    def test_no_lists(self, block):
        if False:
            return 10
        assert_equal(block(1), np.array(1))
        assert_equal(block(np.eye(3)), np.eye(3))

    def test_invalid_nesting(self, block):
        if False:
            for i in range(10):
                print('nop')
        msg = 'depths are mismatched'
        assert_raises_regex(ValueError, msg, block, [1, [2]])
        assert_raises_regex(ValueError, msg, block, [1, []])
        assert_raises_regex(ValueError, msg, block, [[1], 2])
        assert_raises_regex(ValueError, msg, block, [[], 2])
        assert_raises_regex(ValueError, msg, block, [[[1], [2]], [[3, 4]], [5]])

    def test_empty_lists(self, block):
        if False:
            print('Hello World!')
        assert_raises_regex(ValueError, 'empty', block, [])
        assert_raises_regex(ValueError, 'empty', block, [[]])
        assert_raises_regex(ValueError, 'empty', block, [[1], []])

    def test_tuple(self, block):
        if False:
            print('Hello World!')
        assert_raises_regex(TypeError, 'tuple', block, ([1, 2], [3, 4]))
        assert_raises_regex(TypeError, 'tuple', block, [(1, 2), (3, 4)])

    def test_different_ndims(self, block):
        if False:
            while True:
                i = 10
        a = 1.0
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 1, 3))
        result = block([a, b, c])
        expected = np.array([[[1.0, 2.0, 2.0, 3.0, 3.0, 3.0]]])
        assert_equal(result, expected)

    def test_different_ndims_depths(self, block):
        if False:
            return 10
        a = 1.0
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 2, 3))
        result = block([[a, b], [c]])
        expected = np.array([[[1.0, 2.0, 2.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]])
        assert_equal(result, expected)

    def test_block_memory_order(self, block):
        if False:
            return 10
        arr_c = np.zeros((3,) * 3, order='C')
        arr_f = np.zeros((3,) * 3, order='F')
        b_c = [[[arr_c, arr_c], [arr_c, arr_c]], [[arr_c, arr_c], [arr_c, arr_c]]]
        b_f = [[[arr_f, arr_f], [arr_f, arr_f]], [[arr_f, arr_f], [arr_f, arr_f]]]
        assert block(b_c).flags['C_CONTIGUOUS']
        assert block(b_f).flags['F_CONTIGUOUS']
        arr_c = np.zeros((3, 3), order='C')
        arr_f = np.zeros((3, 3), order='F')
        b_c = [[arr_c, arr_c], [arr_c, arr_c]]
        b_f = [[arr_f, arr_f], [arr_f, arr_f]]
        assert block(b_c).flags['C_CONTIGUOUS']
        assert block(b_f).flags['F_CONTIGUOUS']
if __name__ == '__main__':
    run_tests()