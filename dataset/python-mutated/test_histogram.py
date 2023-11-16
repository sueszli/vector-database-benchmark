import sys
import unittest
import numpy
import pytest
import cupy
from cupy import testing
_all_types = (numpy.float16, numpy.float32, numpy.float64, numpy.int8, numpy.int16, numpy.int32, numpy.uint8, numpy.uint16, numpy.bool_)
_signed_types = (numpy.int8, numpy.int16, numpy.int32, numpy.bool_)
if sys.maxsize > 2 ** 32:
    _all_types = _all_types + (numpy.int64, numpy.uint32)
    _signed_types = _signed_types + (numpy.int64,)

def for_all_dtypes_bincount(name='dtype'):
    if False:
        return 10
    return testing.for_dtypes(_all_types, name=name)

def for_signed_dtypes_bincount(name='dtype'):
    if False:
        i = 10
        return i + 15
    return testing.for_dtypes(_signed_types, name=name)

def for_all_dtypes_combination_bincount(names):
    if False:
        for i in range(10):
            print('nop')
    return testing.for_dtypes_combination(_all_types, names=names)

class TestHistogram(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram(self, xp, dtype):
        if False:
            i = 10
            return i + 15
        x = testing.shaped_arange((10,), xp, dtype)
        (y, bin_edges) = xp.histogram(x)
        return (y, bin_edges)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_same_value(self, xp, dtype):
        if False:
            return 10
        x = xp.zeros(10, dtype)
        (y, bin_edges) = xp.histogram(x, 3)
        return (y, bin_edges)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_density(self, xp, dtype):
        if False:
            print('Hello World!')
        x = testing.shaped_arange((10,), xp, dtype)
        (y, bin_edges) = xp.histogram(x, density=True)
        area = xp.sum(y * xp.diff(bin_edges))
        testing.assert_allclose(area, 1)
        return (y, bin_edges)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_histogram_range_lower_outliers(self, xp, dtype):
        if False:
            return 10
        a = xp.arange(10, dtype=dtype) + 0.5
        (h, b) = xp.histogram(a, range=[0, 9])
        assert int(h.sum()) == 9
        return (h, b)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_histogram_range_upper_outliers(self, xp, dtype):
        if False:
            print('Hello World!')
        a = xp.arange(10, dtype=dtype) + 0.5
        (h, b) = xp.histogram(a, range=[1, 10])
        assert int(h.sum()) == 9
        return (h, b)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_histogram_range_with_density(self, xp, dtype):
        if False:
            for i in range(10):
                print('nop')
        a = xp.arange(10, dtype=dtype) + 0.5
        (h, b) = xp.histogram(a, range=[1, 9], density=True)
        testing.assert_allclose(float((h * xp.diff(b)).sum()), 1)
        return h

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_histogram_range_with_weights_and_density(self, xp, dtype):
        if False:
            print('Hello World!')
        a = xp.arange(10, dtype=dtype) + 0.5
        w = xp.arange(10, dtype=dtype) + 0.5
        (h, b) = xp.histogram(a, range=[1, 9], weights=w, density=True)
        testing.assert_allclose(float((h * xp.diff(b)).sum()), 1)
        return h

    def test_histogram_invalid_range(self):
        if False:
            i = 10
            return i + 15
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.histogram(xp.arange(10), range=[1, 9, 15])

    def test_histogram_invalid_range2(self):
        if False:
            while True:
                i = 10
        for xp in (numpy, cupy):
            with pytest.raises(TypeError):
                xp.histogram(xp.arange(10), range=10)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    def test_histogram_weights_mismatch(self, dtype):
        if False:
            return 10
        for xp in (numpy, cupy):
            a = xp.arange(10, dtype=dtype) + 0.5
            w = xp.arange(11, dtype=dtype) + 0.5
            with pytest.raises(ValueError):
                xp.histogram(a, range=[1, 9], weights=w, density=True)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_histogram_int_weights_dtype(self, xp, dtype):
        if False:
            while True:
                i = 10
        a = xp.arange(10, dtype=dtype)
        (h, b) = xp.histogram(a, weights=xp.ones(10, int))
        assert xp.issubdtype(h.dtype, xp.integer)
        return h

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_histogram_float_weights_dtype(self, xp, dtype):
        if False:
            print('Hello World!')
        a = xp.arange(10, dtype=dtype)
        (h, b) = xp.histogram(a, weights=xp.ones(10, float))
        assert xp.issubdtype(h.dtype, xp.floating)
        return h

    def test_histogram_weights_basic(self):
        if False:
            while True:
                i = 10
        v = cupy.random.rand(100)
        w = cupy.ones(100) * 5
        (a, b) = cupy.histogram(v)
        (na, nb) = cupy.histogram(v, density=True)
        (wa, wb) = cupy.histogram(v, weights=w)
        (nwa, nwb) = cupy.histogram(v, weights=w, density=True)
        testing.assert_array_almost_equal(a * 5, wa)
        testing.assert_array_almost_equal(na, nwa)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_histogram_float_weights(self, xp, dtype):
        if False:
            i = 10
            return i + 15
        v = xp.linspace(0, 10, 10, dtype=dtype)
        w = xp.concatenate((xp.zeros(5, dtype=dtype), xp.ones(5, dtype=dtype)))
        (wa, wb) = xp.histogram(v, bins=xp.arange(11), weights=w)
        testing.assert_array_almost_equal(wa, w)
        return wb

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal(type_check=False)
    def test_histogram_int_weights(self, xp, dtype):
        if False:
            i = 10
            return i + 15
        v = xp.asarray([1, 2, 2, 4], dtype=dtype)
        w = xp.asarray([4, 3, 2, 1], dtype=dtype)
        (wa, wb) = xp.histogram(v, bins=4, weights=w)
        testing.assert_array_equal(wa, [4, 5, 0, 1])
        return (wa, wb)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_histogram_int_weights_normalized(self, xp, dtype):
        if False:
            i = 10
            return i + 15
        v = xp.asarray([1, 2, 2, 4], dtype=dtype)
        w = xp.asarray([4, 3, 2, 1], dtype=dtype)
        (wa, wb) = xp.histogram(v, bins=4, weights=w, density=True)
        testing.assert_array_almost_equal(wa, xp.asarray([4, 5, 0, 1]) / 10.0 / 3.0 * 4)
        return wb

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_int_weights_nonuniform_bins(self, xp, dtype):
        if False:
            while True:
                i = 10
        (a, b) = xp.histogram(xp.arange(9, dtype=dtype), xp.asarray([0, 1, 3, 6, 10], dtype=dtype), weights=xp.asarray([2, 1, 1, 1, 1, 1, 1, 1, 1], dtype=dtype), density=True)
        testing.assert_array_almost_equal(a, [0.2, 0.1, 0.1, 0.075])
        return (a, b)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_array_equal(type_check=False)
    def test_histogram_complex_weights(self, xp, dtype):
        if False:
            while True:
                i = 10
        values = xp.asarray([1.3, 2.5, 2.3])
        weights = xp.asarray([1, -1, 2]) + 1j * xp.asarray([2, 1, 2])
        weights = weights.astype(dtype)
        (a, b) = xp.histogram(values, bins=2, weights=weights)
        return (a, b)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_array_equal(type_check=False)
    def test_histogram_complex_weights_uneven_bins(self, xp, dtype):
        if False:
            for i in range(10):
                print('nop')
        values = xp.asarray([1.3, 2.5, 2.3])
        weights = xp.asarray([1, -1, 2]) + 1j * xp.asarray([2, 1, 2])
        weights = weights.astype(dtype)
        (a, b) = xp.histogram(values, bins=xp.asarray([0, 2, 3]), weights=weights)
        return (a, b)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_empty(self, xp, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = xp.array([], dtype)
        (y, bin_edges) = xp.histogram(x)
        return (y, bin_edges)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_int_bins(self, xp, dtype):
        if False:
            return 10
        x = testing.shaped_arange((10,), xp, dtype)
        (y, bin_edges) = xp.histogram(x, 4)
        return (y, bin_edges)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_array_bins(self, xp, dtype):
        if False:
            i = 10
            return i + 15
        x = testing.shaped_arange((10,), xp, dtype)
        bins = testing.shaped_arange((3,), xp, dtype)
        (y, bin_edges) = xp.histogram(x, bins)
        return (y, bin_edges)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_numpy_bins(self, xp, dtype):
        if False:
            return 10
        x = testing.shaped_arange((10,), xp, dtype)
        bins = testing.shaped_arange((3,), numpy, dtype)
        (y, bin_edges) = xp.histogram(x, bins)
        return (y, bin_edges)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_list_bins(self, xp, dtype):
        if False:
            i = 10
            return i + 15
        x = testing.shaped_arange((10,), xp, dtype)
        bins = list(testing.shaped_arange((3,), numpy, dtype))
        (y, bin_edges) = xp.histogram(x, bins)
        return (y, bin_edges)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    def test_histogram_bins_not_ordered(self, dtype):
        if False:
            print('Hello World!')
        for xp in (numpy, cupy):
            x = testing.shaped_arange((10,), xp, dtype)
            bins = xp.array([1, 3, 2], dtype)
            with pytest.raises(ValueError):
                xp.histogram(x, bins)

    @for_all_dtypes_bincount()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_bincount(self, xp, dtype):
        if False:
            while True:
                i = 10
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.bincount(x)

    @for_all_dtypes_bincount()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_bincount_duplicated_value(self, xp, dtype):
        if False:
            return 10
        x = xp.array([1, 2, 2, 1, 2, 4], dtype)
        return xp.bincount(x)

    @for_all_dtypes_combination_bincount(names=['x_type', 'w_type'])
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_bincount_with_weight(self, xp, x_type, w_type):
        if False:
            i = 10
            return i + 15
        x = testing.shaped_arange((3,), xp, x_type)
        w = testing.shaped_arange((3,), xp, w_type)
        return xp.bincount(x, weights=w)

    @for_all_dtypes_bincount()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_bincount_with_minlength(self, xp, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.bincount(x, minlength=5)

    @for_all_dtypes_combination_bincount(names=['x_type', 'w_type'])
    def test_bincount_invalid_weight_length(self, x_type, w_type):
        if False:
            print('Hello World!')
        for xp in (numpy, cupy):
            x = testing.shaped_arange((1,), xp, x_type)
            w = testing.shaped_arange((2,), xp, w_type)
            with pytest.raises((ValueError, TypeError)):
                xp.bincount(x, weights=w)

    @for_signed_dtypes_bincount()
    def test_bincount_negative(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3,), xp, dtype) - 2
            with pytest.raises(ValueError):
                xp.bincount(x)

    @for_all_dtypes_bincount()
    def test_bincount_too_deep(self, dtype):
        if False:
            print('Hello World!')
        for xp in (numpy, cupy):
            x = xp.array([[1]], dtype)
            with pytest.raises(ValueError):
                xp.bincount(x)

    @for_all_dtypes_bincount()
    def test_bincount_too_small(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        for xp in (numpy, cupy):
            x = xp.zeros((), dtype)
            with pytest.raises(ValueError):
                xp.bincount(x)

    @for_all_dtypes_bincount()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_bincount_zero(self, xp, dtype):
        if False:
            i = 10
            return i + 15
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.bincount(x, minlength=0)

    @for_all_dtypes_bincount()
    def test_bincount_too_small_minlength(self, dtype):
        if False:
            while True:
                i = 10
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3,), xp, dtype)
            with pytest.raises((ValueError, TypeError)):
                xp.bincount(x, minlength=-1)

@testing.parameterize(*testing.product({'bins': [[1.5, 2.5, 4.0, 6.0], [-1.0, 1.0, 2.5, 4.0, 20.0], [0.0, 1.0, 1.0, 4.0, 4.0, 10.0]], 'increasing': [True, False], 'right': [True, False], 'shape': [(), (10,), (6, 3, 3)]}))
class TestDigitize:

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_digitize(self, xp, dtype):
        if False:
            return 10
        x = testing.shaped_arange(self.shape, xp, dtype)
        bins = self.bins
        if not self.increasing:
            bins = bins[::-1]
        bins = xp.array(bins)
        y = xp.digitize(x, bins, right=self.right)
        return (y,)

@testing.parameterize({'right': True}, {'right': False})
class TestDigitizeNanInf(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_digitize_nan(self, xp):
        if False:
            print('Hello World!')
        x = testing.shaped_arange((14,), xp, xp.float32)
        x[5] = float('nan')
        bins = xp.array([1.0, 3.0, 5.0, 8.0, 12.0], xp.float32)
        y = xp.digitize(x, bins, right=self.right)
        return (y,)

    @testing.numpy_cupy_array_equal()
    def test_digitize_nan_bins(self, xp):
        if False:
            while True:
                i = 10
        x = testing.shaped_arange((14,), xp, xp.float32)
        bins = xp.array([1.0, 3.0, 5.0, 8.0, float('nan')], xp.float32)
        y = xp.digitize(x, bins, right=self.right)
        return (y,)

    @testing.numpy_cupy_array_equal()
    def test_digitize_nan_bins_repeated(self, xp):
        if False:
            return 10
        x = testing.shaped_arange((14,), xp, xp.float32)
        x[5] = float('nan')
        bins = [1.0, 3.0, 5.0, 8.0, float('nan'), float('nan')]
        bins = xp.array(bins, xp.float32)
        y = xp.digitize(x, bins, right=self.right)
        return (y,)

    @testing.numpy_cupy_array_equal()
    def test_digitize_nan_bins_decreasing(self, xp):
        if False:
            i = 10
            return i + 15
        x = testing.shaped_arange((14,), xp, xp.float32)
        x[5] = float('nan')
        bins = [float('nan'), 8.0, 5.0, 3.0, 1.0]
        bins = xp.array(bins, xp.float32)
        y = xp.digitize(x, bins, right=self.right)
        return (y,)

    @testing.numpy_cupy_array_equal()
    def test_digitize_nan_bins_decreasing_repeated(self, xp):
        if False:
            i = 10
            return i + 15
        x = testing.shaped_arange((14,), xp, xp.float32)
        x[5] = float('nan')
        bins = [float('nan'), float('nan'), float('nan'), 5.0, 3.0, 1.0]
        bins = xp.array(bins, xp.float32)
        y = xp.digitize(x, bins, right=self.right)
        return (y,)

    @testing.numpy_cupy_array_equal()
    def test_digitize_all_nan_bins(self, xp):
        if False:
            print('Hello World!')
        x = testing.shaped_arange((14,), xp, xp.float32)
        x[5] = float('nan')
        bins = [float('nan'), float('nan'), float('nan'), float('nan')]
        bins = xp.array(bins, xp.float32)
        y = xp.digitize(x, bins, right=self.right)
        return (y,)

    @testing.numpy_cupy_array_equal()
    def test_searchsorted_inf(self, xp):
        if False:
            return 10
        x = testing.shaped_arange((14,), xp, xp.float64)
        x[5] = float('inf')
        bins = xp.array([0, 1, 2, 4, 10])
        y = xp.digitize(x, bins, right=self.right)
        return (y,)

    @testing.numpy_cupy_array_equal()
    def test_searchsorted_minf(self, xp):
        if False:
            print('Hello World!')
        x = testing.shaped_arange((14,), xp, xp.float64)
        x[5] = float('-inf')
        bins = xp.array([0, 1, 2, 4, 10])
        y = xp.digitize(x, bins, right=self.right)
        return (y,)

class TestDigitizeInvalid(unittest.TestCase):

    def test_digitize_complex(self):
        if False:
            return 10
        for xp in (numpy, cupy):
            x = testing.shaped_arange((14,), xp, complex)
            bins = xp.array([1.0, 3.0, 5.0, 8.0, 12.0], complex)
            with pytest.raises(TypeError):
                xp.digitize(x, bins)

    def test_digitize_nd_bins(self):
        if False:
            return 10
        for xp in (numpy, cupy):
            x = testing.shaped_arange((14,), xp, xp.float64)
            bins = xp.array([[1], [2]])
            with pytest.raises(ValueError):
                xp.digitize(x, bins)

@testing.parameterize(*testing.product({'weights': [None, 1, 2], 'weights_dtype': [numpy.int32, numpy.float64], 'density': [True, False], 'bins': [10, (8, 16, 12), (16, 8, 12), (16, 12, 8), (12, 8, 16), 'array_list'], 'range': [None, ((20, 50), (10, 100), (0, 40))]}))
class TestHistogramdd:

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-07, rtol=1e-07)
    def test_histogramdd(self, xp, dtype):
        if False:
            i = 10
            return i + 15
        x = testing.shaped_random((100, 3), xp, dtype, scale=100)
        if self.bins == 'array_list':
            bins = [xp.arange(0, 100, 4), xp.arange(0, 100, 10), xp.arange(25)]
        else:
            bins = self.bins
        if self.weights is not None:
            weights = xp.ones((x.shape[0],), dtype=self.weights_dtype)
        else:
            weights = None
        (y, bin_edges) = xp.histogramdd(x, bins=bins, range=self.range, weights=weights, density=self.density)
        return [y] + [e for e in bin_edges]

class TestHistogramddErrors(unittest.TestCase):

    def test_histogramdd_invalid_bins(self):
        if False:
            for i in range(10):
                print('nop')
        for xp in (numpy, cupy):
            x = testing.shaped_random((16, 2), xp, scale=100)
            bins = [xp.arange(0, 100, 10)] * 3
            with pytest.raises(ValueError):
                (y, bin_edges) = xp.histogramdd(x, bins)

    def test_histogramdd_invalid_bins2(self):
        if False:
            return 10
        for xp in (numpy, cupy):
            x = testing.shaped_random((16, 2), xp, scale=100)
            with pytest.raises(ValueError):
                (y, bin_edges) = xp.histogramdd(x, bins=0)

    def test_histogramdd_invalid_bins3(self):
        if False:
            for i in range(10):
                print('nop')
        for xp in (numpy, cupy):
            x = testing.shaped_random((16, 2), xp, scale=100)
            bins = xp.arange(100)
            bins[30] = 99
            with pytest.raises(ValueError):
                (y, bin_edges) = xp.histogramdd(x, bins=bins)

    def test_histogramdd_invalid_bins4(self):
        if False:
            return 10
        for xp in (numpy, cupy):
            x = testing.shaped_random((16, 2), xp, scale=100)
            bins = xp.arange(64).reshape((8, 8))
            with pytest.raises(ValueError):
                (y, bin_edges) = xp.histogramdd(x, bins=bins)

    def test_histogramdd_invalid_range(self):
        if False:
            for i in range(10):
                print('nop')
        for xp in (numpy, cupy):
            x = testing.shaped_random((16, 2), xp, scale=100)
            r = ((0, 100),) * 3
            with pytest.raises(ValueError):
                (y, bin_edges) = xp.histogramdd(x, range=r)

    def test_histogramdd_disallow_arraylike_bins(self):
        if False:
            return 10
        x = testing.shaped_random((16, 2), cupy, scale=100)
        bins = [[0, 10, 20, 50, 90]] * 2
        with pytest.raises(ValueError):
            (y, bin_edges) = cupy.histogramdd(x, bins=bins)

@testing.parameterize(*testing.product({'weights': [None, 1, 2], 'weights_dtype': [numpy.int32, numpy.float64], 'density': [True, False], 'bins': [10, (8, 16), (16, 8), 'array_list', 'array'], 'range': [None, ((20, 50), (10, 100))]}))
class TestHistogram2d:

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-07, rtol=1e-07)
    def test_histogram2d(self, xp, dtype):
        if False:
            while True:
                i = 10
        x = testing.shaped_random((100,), xp, dtype, scale=100)
        y = testing.shaped_random((100,), xp, dtype, scale=100)
        if self.bins == 'array_list':
            bins = [xp.arange(0, 100, 4), xp.arange(0, 100, 10)]
        elif self.bins == 'array':
            bins = xp.arange(0, 100, 4)
        else:
            bins = self.bins
        if self.weights is not None:
            weights = xp.ones((x.shape[0],), dtype=self.weights_dtype)
        else:
            weights = None
        (y, edges0, edges1) = xp.histogram2d(x, y, bins=bins, range=self.range, weights=weights, density=self.density)
        return (y, edges0, edges1)

class TestHistogram2dErrors(unittest.TestCase):

    def test_histogram2d_disallow_arraylike_bins(self):
        if False:
            while True:
                i = 10
        x = testing.shaped_random((16,), cupy, scale=100)
        y = testing.shaped_random((16,), cupy, scale=100)
        bins = [0, 10, 20, 50, 90]
        with pytest.raises(ValueError):
            (y, bin_edges) = cupy.histogram2d(x, y, bins=bins)