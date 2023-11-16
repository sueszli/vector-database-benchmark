import pytest
import cupy
from cupy.cuda import driver
from cupy.cuda import runtime
from cupy import testing
import cupyx.scipy.signal
import numpy as np
try:
    import scipy
    import scipy.signal
except ImportError:
    pass

def _gen_gaussians(xp, center_locs, sigmas, total_length):
    if False:
        for i in range(10):
            print('nop')
    xdata = xp.arange(0, total_length).astype(float)
    out_data = xp.zeros(total_length, dtype=float)
    for (ind, sigma) in enumerate(sigmas):
        tmp = (xdata - center_locs[ind]) / sigma
        out_data += xp.exp(-tmp ** 2)
    return out_data

def _gen_gaussians_even(xp, sigmas, total_length):
    if False:
        i = 10
        return i + 15
    num_peaks = len(sigmas)
    delta = total_length / (num_peaks + 1)
    center_locs = xp.linspace(delta, total_length - delta, num=num_peaks).astype(int)
    out_data = _gen_gaussians(xp, center_locs, sigmas, total_length)
    return (out_data, center_locs)

@pytest.mark.xfail(runtime.is_hip and driver.get_build_version() < 50000000, reason='name_expressions with ROCm 4.3 may not work')
@testing.with_requires('scipy')
class TestPeakProminences:

    @pytest.mark.parametrize('x', [[1, 2, 3], []])
    @testing.numpy_cupy_allclose(scipy_name='scp', type_check=False)
    def test_empty(self, x, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if an empty array is returned if no peaks are provided.\n        '
        out = scp.signal.peak_prominences(x, [])
        return out

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            while True:
                i = 10
        '\n        Test if height of prominences is correctly calculated in signal with\n        rising baseline (peak widths are 1 sample).\n        '
        x = xp.array([-1, 1.2, 1.2, 1, 3.2, 1.3, 2.88, 2.1])
        peaks = xp.array([1, 2, 4, 6])
        out = scp.signal.peak_prominences(x, peaks)
        return out

    @pytest.mark.parametrize('x', [[0.0, 2, 1, 2, 1, 2, 0], [0, 1.0, 0, 1, 0, 1, 0]])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_edge_cases(self, x, xp, scp):
        if False:
            return 10
        '\n        Test edge cases.\n        '
        x = xp.asarray(x)
        peaks = xp.asarray([1, 3, 5])
        out = scp.signal.peak_prominences(x, peaks)
        return out

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_non_contiguous(self, xp, scp):
        if False:
            while True:
                i = 10
        '\n        Test with non-C-contiguous input arrays.\n        '
        x = xp.repeat(xp.asarray([-9, 9, 9, 0, 3, 1.0]), 2)
        peaks = xp.repeat(xp.asarray([1, 2, 4]), 2)
        out = scp.signal.peak_prominences(x[::2], peaks[::2])
        return out

    @pytest.mark.parametrize('wlen', [8, 7, 6, 5, 3.2, 3, 1.1])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_wlen(self, wlen, xp, scp):
        if False:
            i = 10
            return i + 15
        '\n        Test if wlen actually shrinks the evaluation range correctly.\n        '
        x = xp.asarray([0, 1, 2, 3.0, 1, 0, -1])
        peak = xp.asarray([3])
        return scp.signal.peak_prominences(x, peak, wlen)

    def test_exceptions(self):
        if False:
            return 10
        '\n        Verify that exceptions and warnings are raised.\n        '
        for (xp, scp) in [(cupy, cupyx.scipy), (np, scipy)]:
            with pytest.raises(ValueError, match='1-D array'):
                scp.signal.peak_prominences([[0, 1, 1, 0]], [1, 2])
            with pytest.raises(ValueError, match='1-D array'):
                scp.signal.peak_prominences([0, 1, 1, 0], [[1, 2]])
            with pytest.raises(ValueError, match='1-D array'):
                scp.signal.peak_prominences(3, [0])
            with pytest.raises(ValueError, match='not a valid index'):
                scp.signal.peak_prominences([], [0])
            for p in [-100, -1, 3, 1000]:
                with pytest.raises(ValueError, match='not a valid index'):
                    scp.signal.peak_prominences([1, 0, 2], [p])
            with pytest.raises(ValueError, match='wlen'):
                scp.signal.peak_prominences(xp.arange(10), [3, 5], wlen=1)

@pytest.mark.xfail(runtime.is_hip and driver.get_build_version() < 50000000, reason='name_expressions with ROCm 4.3 may not work')
@testing.with_requires('scipy')
class TestPeakWidths:

    @pytest.mark.parametrize('x', [[1, 2, 3], []])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_empty(self, x, xp, scp):
        if False:
            i = 10
            return i + 15
        '\n        Test if an empty array is returned if no peaks are provided.\n        '
        widths = scp.signal.peak_widths(x, [])
        return widths

    @pytest.mark.filterwarnings('ignore:some peaks have a width of 0')
    @pytest.mark.parametrize('rel_height', [0, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, rel_height, xp, scp):
        if False:
            return 10
        '\n        Test a simple use case with easy to verify results at different\n        relative heights.\n        '
        x = xp.array([1, 0, 1, 2, 1, 0, -1])
        out = scp.signal.peak_widths(x, [3], rel_height)
        return out

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_non_contiguous(self, xp, scp):
        if False:
            return 10
        '\n        Test with non-C-contiguous input arrays.\n        '
        x = xp.repeat(xp.asarray([0, 100, 50]), 4)
        peaks = xp.repeat(xp.asarray([1]), 3)
        result = scp.signal.peak_widths(x[::4], peaks[::3])
        return result

    def test_exceptions(self):
        if False:
            while True:
                i = 10
        '\n        Verify that argument validation works as intended.\n        '
        for (xp, scp) in [(cupy, cupyx.scipy), (np, scipy)]:
            with pytest.raises(ValueError, match='1-D array'):
                scp.signal.peak_widths(xp.zeros((3, 4)), xp.ones(3))
            with pytest.raises(ValueError, match='1-D array'):
                scp.signal.peak_widths(3, [0])
            with pytest.raises(ValueError, match='1-D array'):
                scp.signal.peak_widths(xp.arange(10), xp.ones((3, 2), dtype=xp.intp))
            with pytest.raises(ValueError, match='1-D array'):
                scp.signal.peak_widths(xp.arange(10), 3)
            with pytest.raises(ValueError, match='not a valid index'):
                scp.signal.peak_widths(xp.arange(10), [8, 11])
            with pytest.raises(ValueError, match='not a valid index'):
                scp.signal.peak_widths([], [1, 2])
            with pytest.raises(ValueError, match='rel_height'):
                scp.signal.peak_widths([0, 1, 0, 1, 0], [1, 3], rel_height=-1)
            with pytest.raises(TypeError, match='None'):
                scp.signal.peak_widths([1, 2, 1], [1], prominence_data=(None, None, None))

    def test_mismatching_prominence_data(self):
        if False:
            i = 10
            return i + 15
        'Test with mismatching peak and / or prominence data.'
        for (xp, scp) in [(cupy, cupyx.scipy), (np, scipy)]:
            x = xp.asarray([0, 1, 0])
            peak = [1]
            for (i, (prominences, left_bases, right_bases)) in enumerate([((1.0,), (-1,), (2,)), ((1.0,), (0,), (3,)), ((1.0,), (2,), (0,)), ((1.0, 1.0), (0, 0), (2, 2)), ((1.0, 1.0), (0,), (2,)), ((1.0,), (0, 0), (2,)), ((1.0,), (0,), (2, 2))]):
                prominence_data = (xp.array(prominences, dtype=xp.float64), xp.array(left_bases, dtype=xp.intp), xp.array(right_bases, dtype=xp.intp))
                if i < 3:
                    match = 'prominence data is invalid'
                else:
                    match = 'arrays in `prominence_data` must have the same shape'
                with pytest.raises(ValueError, match=match):
                    scp.signal.peak_widths(x, peak, prominence_data=prominence_data)

    @pytest.mark.filterwarnings('ignore:some peaks have a width of 0')
    @pytest.mark.parametrize('rel_height', [0, 2 / 3])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_intersection_rules(self, rel_height, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        'Test if x == eval_height counts as an intersection.'
        x = [0, 1, 2, 1, 3, 3, 3, 1, 2, 1, 0]
        out = scp.signal.peak_widths(x, peaks=[5], rel_height=rel_height)
        return out

@pytest.mark.xfail(runtime.is_hip and driver.get_build_version() < 50000000, reason='name_expressions with ROCm 4.3 may not work')
@testing.with_requires('scipy')
class TestFindPeaks:
    property_keys = {'peak_heights', 'left_thresholds', 'right_thresholds', 'prominences', 'left_bases', 'right_bases', 'widths', 'width_heights', 'left_ips', 'right_ips'}

    @testing.numpy_cupy_allclose(scipy_name='scp', type_check=False)
    def test_constant(self, xp, scp):
        if False:
            return 10
        '\n        Test behavior for signal without local maxima.\n        '
        open_interval = (None, None)
        (peaks, props) = scp.signal.find_peaks(xp.ones(10), height=open_interval, threshold=open_interval, prominence=open_interval, width=open_interval)
        return (peaks,) + tuple([props[k] for k in self.property_keys if k in props])

    @pytest.mark.parametrize('plateau_size', [(None, None), 4, (None, 3.5), (5, 50)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_plateau_size(self, plateau_size, xp, scp):
        if False:
            print('Hello World!')
        '\n        Test plateau size condition for peaks.\n        '
        plateau_sizes = xp.array([1, 2, 3, 4, 8, 20, 111])
        x = xp.zeros(plateau_sizes.size * 2 + 1)
        x[1::2] = plateau_sizes
        repeats = xp.ones(x.size, dtype=int)
        repeats[1::2] = x[1::2]
        x = xp.repeat(x, repeats.tolist())
        (peaks, props) = scp.signal.find_peaks(x, plateau_size=plateau_size)
        return (peaks,) + tuple([props[k] for k in self.property_keys if k in props])

    @pytest.mark.parametrize('height', [(None, None), 0.5, (None, 3), (2, 3)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_height_condition(self, height, xp, scp):
        if False:
            return 10
        '\n        Test height condition for peaks.\n        '
        x = xp.asarray([0.0, 1 / 3, 0.0, 2.5, 0, 4.0, 0])
        (peaks, props) = scp.signal.find_peaks(x, height=height)
        return (peaks,) + tuple([props[k] for k in self.property_keys if k in props])

    @pytest.mark.parametrize('threshold', [(None, None), 2, 3.5, (None, 5), (None, 4), (2, 4)])
    @testing.numpy_cupy_allclose(scipy_name='scp', type_check=False)
    def test_threshold_condition(self, threshold, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test threshold condition for peaks.\n        '
        x = xp.asarray([0, 2, 1, 4, -1])
        (peaks, props) = scp.signal.find_peaks(x, threshold=threshold)
        return (peaks,) + tuple([props[k] for k in self.property_keys if k in props])

    @pytest.mark.parametrize('distance', [3, 3.0001])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_distance_condition(self, distance, xp, scp):
        if False:
            print('Hello World!')
        '\n        Test distance condition for peaks.\n        '
        peaks_all = xp.arange(1, 21, 3)
        x = xp.zeros(21)
        x[peaks_all] += xp.linspace(1, 2, peaks_all.size)
        (peaks, props) = scp.signal.find_peaks(x, distance=distance)
        return (peaks,) + tuple([props[k] for k in self.property_keys if k in props])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_distance_priority(self, xp, scp):
        if False:
            return 10
        x = xp.asarray([-2, 1, -1, 0, -3])
        (peaks, props) = scp.signal.find_peaks(x, distance=10)
        return (peaks,) + tuple([props[k] for k in self.property_keys if k in props])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_prominence_condition(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test prominence condition for peaks.\n        '
        x = xp.linspace(0, 10, 100)
        peaks_true = xp.arange(1, 99, 2)
        offset = xp.linspace(1, 10, peaks_true.size)
        x[peaks_true] += offset
        interval = (3, 9)
        (peaks, props) = scp.signal.find_peaks(x, prominence=interval)
        return (peaks,) + tuple([props[k] for k in self.property_keys if k in props])

    @testing.numpy_cupy_allclose(scipy_name='scp', type_check=False)
    def test_width_condition(self, xp, scp):
        if False:
            return 10
        '\n        Test width condition for peaks.\n        '
        x = xp.array([1, 0, 1, 2, 1, 0, -1, 4, 0])
        (peaks, props) = scp.signal.find_peaks(x, width=(None, 2), rel_height=0.75)
        return (peaks,) + tuple([props[k] for k in self.property_keys if k in props])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_properties(self, xp, scp):
        if False:
            i = 10
            return i + 15
        '\n        Test returned properties.\n        '
        open_interval = (None, None)
        x = xp.asarray([0, 1, 0, 2, 1.5, 0, 3, 0, 5, 9])
        (peaks, props) = scp.signal.find_peaks(x, height=open_interval, threshold=open_interval, prominence=open_interval, width=open_interval)
        return (peaks,) + tuple([props[k] for k in self.property_keys if k in props])

    def test_raises(self):
        if False:
            while True:
                i = 10
        '\n        Test exceptions raised by function.\n        '
        for (xp, scp) in [(cupy, cupyx.scipy), (np, scipy)]:
            with pytest.raises(ValueError, match='1-D array'):
                scp.signal.find_peaks(xp.array(1))
            with pytest.raises(ValueError, match='1-D array'):
                scp.signal.find_peaks(xp.ones((2, 2)))
            with pytest.raises(ValueError, match='distance'):
                scp.signal.find_peaks(xp.arange(10), distance=-1)

    @pytest.mark.filterwarnings('ignore:some peaks have a prominence of 0', 'ignore:some peaks have a width of 0')
    @testing.numpy_cupy_allclose(scipy_name='scp', type_check=False)
    def test_wlen_smaller_plateau(self, xp, scp):
        if False:
            i = 10
            return i + 15
        "\n        Test behavior of prominence and width calculation if the given window\n        length is smaller than a peak's plateau size.\n\n        Regression test for gh-9110.\n        "
        (peaks, props) = scp.signal.find_peaks([0, 1, 1, 1, 0], prominence=(None, None), width=(None, None), wlen=2)
        return (peaks,) + tuple([props[k] for k in self.property_keys if k in props])

@pytest.mark.xfail(runtime.is_hip and driver.get_build_version() < 50000000, reason='name_expressions with ROCm 4.3 may not work')
@testing.with_requires('scipy')
class TestArgrel:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_empty(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        z1 = xp.zeros(5)
        i = scp.signal.argrelmin(z1)
        z2 = xp.zeros((3, 5))
        (row1, col1) = scp.signal.argrelmin(z2, axis=0)
        (row2, col2) = scp.signal.argrelmin(z2, axis=1)
        return (i[0], row1, col1, row2, col2)

    @pytest.mark.parametrize('func_name', ['argrelmax', 'argrelmin'])
    @pytest.mark.parametrize('axis', [0, 1])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, func_name, axis, xp, scp):
        if False:
            i = 10
            return i + 15
        x = xp.array([[1, 2, 2, 3, 2], [2, 1, 2, 2, 3], [3, 2, 1, 2, 2], [2, 3, 2, 1, 2], [1, 2, 3, 2, 1]])
        func = getattr(scp.signal, func_name)
        (row, col) = func(x, axis=axis)
        order = xp.argsort(row)
        return (row[order], col[order])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highorder(self, xp, scp):
        if False:
            while True:
                i = 10
        order = 2
        sigmas = [1.0, 2.0, 10.0, 5.0, 15.0]
        (test_data, act_locs) = _gen_gaussians_even(xp, sigmas, 500)
        test_data[act_locs + order] = test_data[act_locs] * 0.99999
        test_data[act_locs - order] = test_data[act_locs] * 0.99999
        rel_max_locs = scp.signal.argrelmax(test_data, order=order, mode='clip')[0]
        return rel_max_locs

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_2d_gaussians(self, xp, scp):
        if False:
            return 10
        sigmas = [1.0, 2.0, 10.0]
        (test_data, _) = _gen_gaussians_even(xp, sigmas, 100)
        rot_factor = 20
        rot_range = xp.arange(0, len(test_data)) - rot_factor
        test_data_2 = xp.vstack([test_data, test_data[rot_range]])
        (rel_max_rows, rel_max_cols) = scp.signal.argrelmax(test_data_2, axis=1, order=1)
        return (rel_max_rows, rel_max_cols)