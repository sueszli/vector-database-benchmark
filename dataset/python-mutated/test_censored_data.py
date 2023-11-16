import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from scipy.stats import CensoredData

class TestCensoredData:

    def test_basic(self):
        if False:
            return 10
        uncensored = [1]
        left = [0]
        right = [2, 5]
        interval = [[2, 3]]
        data = CensoredData(uncensored, left=left, right=right, interval=interval)
        assert_equal(data._uncensored, uncensored)
        assert_equal(data._left, left)
        assert_equal(data._right, right)
        assert_equal(data._interval, interval)
        udata = data._uncensor()
        assert_equal(udata, np.concatenate((uncensored, left, right, np.mean(interval, axis=1))))

    def test_right_censored(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.array([0, 3, 2.5])
        is_censored = np.array([0, 1, 0], dtype=bool)
        data = CensoredData.right_censored(x, is_censored)
        assert_equal(data._uncensored, x[~is_censored])
        assert_equal(data._right, x[is_censored])
        assert_equal(data._left, [])
        assert_equal(data._interval, np.empty((0, 2)))

    def test_left_censored(self):
        if False:
            while True:
                i = 10
        x = np.array([0, 3, 2.5])
        is_censored = np.array([0, 1, 0], dtype=bool)
        data = CensoredData.left_censored(x, is_censored)
        assert_equal(data._uncensored, x[~is_censored])
        assert_equal(data._left, x[is_censored])
        assert_equal(data._right, [])
        assert_equal(data._interval, np.empty((0, 2)))

    def test_interval_censored_basic(self):
        if False:
            for i in range(10):
                print('nop')
        a = [0.5, 2.0, 3.0, 5.5]
        b = [1.0, 2.5, 3.5, 7.0]
        data = CensoredData.interval_censored(low=a, high=b)
        assert_array_equal(data._interval, np.array(list(zip(a, b))))
        assert data._uncensored.shape == (0,)
        assert data._left.shape == (0,)
        assert data._right.shape == (0,)

    def test_interval_censored_mixed(self):
        if False:
            return 10
        a = [0.5, -np.inf, -13.0, 2.0, 1.0, 10.0, -1.0]
        b = [0.5, 2500.0, np.inf, 3.0, 1.0, 11.0, np.inf]
        data = CensoredData.interval_censored(low=a, high=b)
        assert_array_equal(data._interval, [[2.0, 3.0], [10.0, 11.0]])
        assert_array_equal(data._uncensored, [0.5, 1.0])
        assert_array_equal(data._left, [2500.0])
        assert_array_equal(data._right, [-13.0, -1.0])

    def test_interval_to_other_types(self):
        if False:
            print('Hello World!')
        interval = np.array([[0, 1], [2, 2], [3, 3], [9, np.inf], [8, np.inf], [-np.inf, 0], [1, 2]])
        data = CensoredData(interval=interval)
        assert_equal(data._uncensored, [2, 3])
        assert_equal(data._left, [0])
        assert_equal(data._right, [9, 8])
        assert_equal(data._interval, [[0, 1], [1, 2]])

    def test_empty_arrays(self):
        if False:
            return 10
        data = CensoredData(uncensored=[], left=[], right=[], interval=[])
        assert data._uncensored.shape == (0,)
        assert data._left.shape == (0,)
        assert data._right.shape == (0,)
        assert data._interval.shape == (0, 2)
        assert len(data) == 0

    def test_invalid_constructor_args(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError, match='must be a one-dimensional'):
            CensoredData(uncensored=[[1, 2, 3]])
        with pytest.raises(ValueError, match='must be a one-dimensional'):
            CensoredData(left=[[1, 2, 3]])
        with pytest.raises(ValueError, match='must be a one-dimensional'):
            CensoredData(right=[[1, 2, 3]])
        with pytest.raises(ValueError, match='must be a two-dimensional'):
            CensoredData(interval=[[1, 2, 3]])
        with pytest.raises(ValueError, match='must not contain nan'):
            CensoredData(uncensored=[1, np.nan, 2])
        with pytest.raises(ValueError, match='must not contain nan'):
            CensoredData(left=[1, np.nan, 2])
        with pytest.raises(ValueError, match='must not contain nan'):
            CensoredData(right=[1, np.nan, 2])
        with pytest.raises(ValueError, match='must not contain nan'):
            CensoredData(interval=[[1, np.nan], [2, 3]])
        with pytest.raises(ValueError, match='both values must not be infinite'):
            CensoredData(interval=[[1, 3], [2, 9], [np.inf, np.inf]])
        with pytest.raises(ValueError, match='left value must not exceed the right'):
            CensoredData(interval=[[1, 0], [2, 2]])

    @pytest.mark.parametrize('func', [CensoredData.left_censored, CensoredData.right_censored])
    def test_invalid_left_right_censored_args(self, func):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError, match='`x` must be one-dimensional'):
            func([[1, 2, 3]], [0, 1, 1])
        with pytest.raises(ValueError, match='`censored` must be one-dimensional'):
            func([1, 2, 3], [[0, 1, 1]])
        with pytest.raises(ValueError, match='`x` must not contain'):
            func([1, 2, np.nan], [0, 1, 1])
        with pytest.raises(ValueError, match='must have the same length'):
            func([1, 2, 3], [0, 0, 1, 1])

    def test_invalid_censored_args(self):
        if False:
            return 10
        with pytest.raises(ValueError, match='`low` must be a one-dimensional'):
            CensoredData.interval_censored(low=[[3]], high=[4, 5])
        with pytest.raises(ValueError, match='`high` must be a one-dimensional'):
            CensoredData.interval_censored(low=[3], high=[[4, 5]])
        with pytest.raises(ValueError, match='`low` must not contain'):
            CensoredData.interval_censored([1, 2, np.nan], [0, 1, 1])
        with pytest.raises(ValueError, match='must have the same length'):
            CensoredData.interval_censored([1, 2, 3], [0, 0, 1, 1])

    def test_count_censored(self):
        if False:
            for i in range(10):
                print('nop')
        x = [1, 2, 3]
        data1 = CensoredData(x)
        assert data1.num_censored() == 0
        data2 = CensoredData(uncensored=[2.5], left=[10], interval=[[0, 1]])
        assert data2.num_censored() == 2