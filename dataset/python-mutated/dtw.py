"""
Dynamic Time Warping (DTW)
--------------------------
"""
import copy
from typing import Callable, Optional, Union
import numpy as np
import pandas as pd
import xarray as xr
from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not
from darts.timeseries import DIMS
from .cost_matrix import CostMatrix
from .window import CRWindow, NoWindow, Window
logger = get_logger(__name__)
SeriesValue = Union[np.ndarray, np.floating]
DistanceFunc = Callable[[SeriesValue, SeriesValue], float]

def _dtw_cost_matrix(x: np.ndarray, y: np.ndarray, dist: DistanceFunc, window: Window) -> CostMatrix:
    if False:
        i = 10
        return i + 15
    dtw = CostMatrix._from_window(window)
    dtw.fill(np.inf)
    dtw[0, 0] = 0
    for (i, j) in window:
        cost = dist(x[i - 1], y[j - 1])
        min_cost_prev = min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
        dtw[i, j] = cost + min_cost_prev
    return dtw

def _dtw_path(dtw: CostMatrix) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    i = dtw.n
    j = dtw.m
    path = []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        stencil = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
        costs = [dtw[i, j] for (i, j) in stencil]
        index_min = costs.index(min(costs))
        (i, j) = stencil[index_min]
    path.reverse()
    return np.array(path)

def _down_sample(high_res: np.ndarray):
    if False:
        for i in range(10):
            print('nop')
    needs_padding = len(high_res) & 1
    if needs_padding:
        high_res = np.append(high_res, high_res[-1])
    low_res = np.reshape(high_res, (-1, 2))
    low_res = np.mean(low_res, axis=1)
    return low_res

def _expand_window(low_res_path: np.ndarray, n: int, m: int, radius: int) -> CRWindow:
    if False:
        print('Hello World!')
    high_res_grid = CRWindow(n, m)

    def is_valid(cell):
        if False:
            return 10
        valid_x = 1 <= cell[0] <= n
        valid_y = 1 <= cell[1] <= m
        return valid_x and valid_y
    pattern = [(0, 0, 2), (1, 0, 3), (2, 1, 2)]
    for (i, j) in low_res_path:
        for (column, start, end) in pattern:
            column += i * 2 + 1
            start = max(1, min(m + 1, start + j * 2 - radius))
            end = max(1, min(m + 1, end + j * 2 + 1 + radius))
            for k in range(0, min(radius + 1, column, n - column + 1)):
                high_res_grid.add_range(column - k, start, end)
    return high_res_grid

def _fast_dtw(x: np.ndarray, y: np.ndarray, dist: DistanceFunc, radius: int, depth: int=0) -> CostMatrix:
    if False:
        i = 10
        return i + 15
    n = len(x)
    m = len(y)
    min_size = radius + 2
    if n < min_size or m < min_size or radius == -1:
        window = NoWindow()
        window.init_size(n, m)
        cost = _dtw_cost_matrix(x, y, dist, window)
        return cost
    half_x = _down_sample(x)
    half_y = _down_sample(y)
    low_res_cost = _fast_dtw(half_x, half_y, dist, radius, depth + 1)
    low_res_path = _dtw_path(low_res_cost)
    window = _expand_window(low_res_path, len(x), len(y), radius)
    cost = _dtw_cost_matrix(x, y, dist, window)
    return cost

def _default_distance_multi(x_values: np.ndarray, y_values: np.ndarray):
    if False:
        i = 10
        return i + 15
    return np.sum(np.abs(x_values - y_values))

def _default_distance_uni(x_value: float, y_value: float):
    if False:
        print('Hello World!')
    return abs(x_value - y_value)

class DTWAlignment:
    """
    Dynamic Time Warping (DTW) Alignment.

    Attributes
    ----------
    n
        The length of `series1`
    m
        The length of `series2`
    series1
        A `TimeSeries` to align with `series2`.
    series2
        A `TimeSeries` to align with `series1`.
    cost
        The `CostMatrix` for DTW.
    """
    n: int
    m: int
    series1: TimeSeries
    series2: TimeSeries
    cost: CostMatrix

    def __init__(self, series1: TimeSeries, series2: TimeSeries, cost: CostMatrix):
        if False:
            i = 10
            return i + 15
        self.n = len(series1)
        self.m = len(series2)
        self.series1 = series1
        self.series2 = series2
        self.cost = cost
    from ._plot import plot, plot_alignment

    def path(self) -> np.ndarray:
        if False:
            print('Hello World!')
        'Gives the index paths from `series1` to `series2`.\n\n        Returns\n        -------\n        np.ndarray of shape `(len(path), 2)`\n            An array of indices [[i0,j0], [i1,j1], [i2,j2], ...], where i indexes into series1\n            and j indexes into series2.\n            Indices are in monotonic order, path[n] >= path[n-1]\n        '
        if hasattr(self, '_path'):
            return self._path
        self._path = _dtw_path(self.cost)
        return self._path

    def distance(self) -> float:
        if False:
            print('Hello World!')
        'Gives the total distance between pair-wise elements in the two series after warping.\n\n        Returns\n        -------\n        float\n            The total distance between pair-wise elements in the two series after warping.\n        '
        return self.cost[self.n, self.m]

    def mean_distance(self) -> float:
        if False:
            return 10
        'Gives the mean distance between pair-wise elements in the two series after warping.\n\n        Returns\n        -------\n        float\n            The mean distance between pair-wise elements in the two series after warping.\n        '
        if hasattr(self, '_mean_distance'):
            return self._mean_distance
        path = self.path()
        self._mean_distance = self.distance() / len(path)
        return self._mean_distance

    def warped(self) -> (TimeSeries, TimeSeries):
        if False:
            return 10
        'Warps the two time series according to the warp path returned by `DTWAlignment.path()`, which minimizes the\n        pair-wise distance.\n        This will bring two time series that are out-of-phase back into phase.\n\n        Returns\n        -------\n        (TimeSeries, TimeSeries)\n            Two new TimeSeries instances of the same length, indexed by pd.RangeIndex.\n        '
        xa1 = self.series1.data_array(copy=False)
        xa2 = self.series2.data_array(copy=False)
        path = self.path()
        (values1, values2) = (xa1.values[path[:, 0]], xa2.values[path[:, 1]])
        warped_series1 = xr.DataArray(data=values1, dims=xa1.dims, coords={self.series1._time_dim: pd.RangeIndex(values1.shape[0]), DIMS[1]: xa1.coords[DIMS[1]]}, attrs=xa1.attrs)
        warped_series2 = xr.DataArray(data=values2, dims=xa2.dims, coords={self.series2._time_dim: pd.RangeIndex(values2.shape[0]), DIMS[1]: xa2.coords[DIMS[1]]}, attrs=xa2.attrs)
        (time_dim1, time_dim2) = (self.series1._time_dim, self.series2._time_dim)
        take_dates = False
        if take_dates:
            time_index = warped_series1[time_dim1]
            time_index = time_index.rename({time_dim1: time_dim2})
            warped_series2[time_dim2] = time_index
        return (TimeSeries.from_xarray(warped_series1), TimeSeries.from_xarray(warped_series2))

def dtw(series1: TimeSeries, series2: TimeSeries, window: Optional[Window]=None, distance: Union[DistanceFunc, None]=None, multi_grid_radius: int=-1) -> DTWAlignment:
    if False:
        while True:
            i = 10
    '\n    Determines the optimal alignment between two time series `series1` and `series2`, according to the Dynamic Time\n    Warping algorithm.\n    The alignment minimizes the distance between pair-wise elements after warping.\n    All elements in the two series are matched and are in strictly monotonically increasing order.\n    Considers only the values in the series, ignoring the time axis.\n\n    Dynamic Time Warping can be applied to determine how closely two time series correspond,\n    irrespective of phase, length or speed differences.\n\n    Parameters\n    ----------\n    series1\n        A `TimeSeries` to align with `series2`.\n    series2\n        A `TimeSeries` to align with `series1`.\n    window\n        Optionally, a `Window` used to constrain the search for the optimal alignment: see `SakoeChiba` and `Itakura`.\n        Default considers all possible alignments (`NoWindow`).\n    distance\n        Function taking as input either two `floats` for univariate series or two `np.ndarray`,\n        and returning the distance between them.\n\n        Defaults to the abs difference for univariate-data and the\n        sum of the abs difference for multi-variate series.\n    multi_grid_radius\n        Default radius of `-1` results in an exact evaluation of the dynamic time warping algorithm.\n        Without constraints DTW runs in O(nxm) time where n,m are the size of the series.\n        Exact evaluation with no constraints, will result in a performance warning on large datasets.\n\n        Setting `multi_grid_radius` to a value other than `-1`, will enable the approximate multi-grid solver,\n        which executes in linear time, vs quadratic time for exact evaluation.\n        Increasing radius trades solution accuracy for performance.\n\n    Returns\n    -------\n    DTWAlignment\n        Helper object for getting warp path, mean_distance, distance and warped time series\n    '
    if window is None:
        window = NoWindow()
    if multi_grid_radius == -1 and type(window) is NoWindow and (len(series1) * len(series2) > 10 ** 6):
        logger.warning('Exact evaluation will result in poor performance on large datasets. Consider enabling multi-grid or using a window.')
    both_univariate = series1.is_univariate and series2.is_univariate
    if distance is None:
        raise_if_not(series1.n_components == series2.n_components, 'Expected series to have same number of components, or to supply custom distance function', logger)
        distance = _default_distance_uni if both_univariate else _default_distance_multi
    if both_univariate:
        values_x = series1.univariate_values(copy=False)
        values_y = series2.univariate_values(copy=False)
    else:
        values_x = series1.values(copy=False)
        values_y = series2.values(copy=False)
    raise_if(np.any(np.isnan(values_x)), 'Dynamic Time Warping does not support nan values. You can use the module darts.utils.missing_values to fill them, before passing them to dtw.', logger)
    raise_if(np.any(np.isnan(values_y)), 'Dynamic Time Warping does not support nan values. You can use the module darts.utils.missing_values to fill them,before passing it into dtw', logger)
    window = copy.deepcopy(window)
    window.init_size(len(values_x), len(values_y))
    raise_if(multi_grid_radius < -1, 'Expected multi-grid radius to be positive or -1')
    if multi_grid_radius >= 0:
        raise_if_not(isinstance(window, NoWindow), 'Multi-grid solver does not currently support windows', logger)
        cost_matrix = _fast_dtw(values_x, values_y, distance, multi_grid_radius)
    else:
        cost_matrix = _dtw_cost_matrix(values_x, values_y, distance, window)
    return DTWAlignment(series1, series2, cost_matrix)