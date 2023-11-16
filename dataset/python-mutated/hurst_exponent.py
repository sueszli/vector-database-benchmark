import numpy as np
try:
    from numba import njit
except ImportError:
    njit = lambda a: a
from jesse.helpers import get_candle_source, slice_candles
from scipy import signal

def hurst_exponent(candles: np.ndarray, min_chunksize: int=8, max_chunksize: int=200, num_chunksize: int=5, method: int=1, source_type: str='close') -> float:
    if False:
        return 10
    '\n    Hurst Exponent\n\n    :param candles: np.ndarray\n    :param min_chunksize: int - default: 8\n    :param max_chunksize: int - default: 200\n    :param num_chunksize: int - default: 5\n    :param method: int - default: 1 - 0: RS | 1: DMA | 2: DSOD\n    :param source_type: str - default: "close"\n\n    :return: float\n    '
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, False)
        source = get_candle_source(candles, source_type=source_type)
    if method == 0:
        h = hurst_rs(np.diff(source), min_chunksize, max_chunksize, num_chunksize)
    elif method == 1:
        h = hurst_dma(source, min_chunksize, max_chunksize, num_chunksize)
    elif method == 2:
        h = hurst_dsod(source)
    else:
        raise NotImplementedError('The method choose is not implemented.')
    return None if np.isnan(h) else h

@njit
def hurst_rs(x, min_chunksize, max_chunksize, num_chunksize):
    if False:
        for i in range(10):
            print('nop')
    'Estimate the Hurst exponent using R/S method.\n    Estimates the Hurst (H) exponent using the R/S method from the time series.\n    The R/S method consists of dividing the series into pieces of equal size\n    `series_len` and calculating the rescaled range. This repeats the process\n    for several `series_len` values and adjusts data regression to obtain the H.\n    `series_len` will take values between `min_chunksize` and `max_chunksize`,\n    the step size from `min_chunksize` to `max_chunksize` can be controlled\n    through the parameter `step_chunksize`.\n    Parameters\n    ----------\n    x : 1D-array\n        A time series to calculate hurst exponent, must have more elements\n        than `min_chunksize` and `max_chunksize`.\n    min_chunksize : int\n        This parameter allow you control the minimum window size.\n    max_chunksize : int\n        This parameter allow you control the maximum window size.\n    num_chunksize : int\n        This parameter allow you control the size of the step from minimum to\n        maximum window size. Bigger step means fewer calculations.\n    out : 1-element-array, optional\n        one element array to store the output.\n    Returns\n    -------\n    H : float\n        A estimation of Hurst exponent.\n    References\n    ----------\n    Hurst, H. E. (1951). Long term storage capacity of reservoirs. ASCE\n    Transactions, 116(776), 770-808.\n    Alessio, E., Carbone, A., Castelli, G. et al. Eur. Phys. J. B (2002) 27:\n    197. http://dx.doi.org/10.1140/epjb/e20020150\n    '
    N = len(x)
    max_chunksize += 1
    rs_tmp = np.empty(N, dtype=np.float64)
    chunk_size_list = np.linspace(min_chunksize, max_chunksize, num_chunksize).astype(np.int64)
    rs_values_list = np.empty(num_chunksize, dtype=np.float64)
    for i in range(num_chunksize):
        chunk_size = chunk_size_list[i]
        number_of_chunks = int(len(x) / chunk_size)
        for idx in range(number_of_chunks):
            ini = idx * chunk_size
            end = ini + chunk_size
            chunk = x[ini:end]
            z = np.cumsum(chunk - np.mean(chunk))
            rs_tmp[idx] = np.divide(np.max(z) - np.min(z), np.nanstd(chunk))
        rs_values_list[i] = np.nanmean(rs_tmp[:idx + 1])
    (H, c) = np.linalg.lstsq(a=np.vstack((np.log(chunk_size_list), np.ones(num_chunksize))).T, b=np.log(rs_values_list))[0]
    return H

def hurst_dma(prices, min_chunksize=8, max_chunksize=200, num_chunksize=5):
    if False:
        while True:
            i = 10
    'Estimate the Hurst exponent using R/S method.\n\n    Estimates the Hurst (H) exponent using the DMA method from the time series.\n    The DMA method consists on calculate the moving average of size `series_len`\n    and subtract it to the original series and calculating the standard\n    deviation of that result. This repeats the process for several `series_len`\n    values and adjusts data regression to obtain the H. `series_len` will take\n    values between `min_chunksize` and `max_chunksize`, the step size from\n    `min_chunksize` to `max_chunksize` can be controlled through the parameter\n    `step_chunksize`.\n\n    Parameters\n    ----------\n    prices\n    min_chunksize\n    max_chunksize\n    num_chunksize\n\n    Returns\n    -------\n    hurst_exponent : float\n        Estimation of hurst exponent.\n\n    References\n    ----------\n    Alessio, E., Carbone, A., Castelli, G. et al. Eur. Phys. J. B (2002) 27:\n    197. https://dx.doi.org/10.1140/epjb/e20020150\n\n    '
    max_chunksize += 1
    N = len(prices)
    n_list = np.arange(min_chunksize, max_chunksize, num_chunksize, dtype=np.int64)
    dma_list = np.empty(len(n_list))
    factor = 1 / (N - max_chunksize)
    for (i, n) in enumerate(n_list):
        b = np.divide([n - 1] + (n - 1) * [-1], n)
        noise = np.power(signal.lfilter(b, 1, prices)[max_chunksize:], 2)
        dma_list[i] = np.sqrt(factor * np.sum(noise))
    (H, const) = np.linalg.lstsq(a=np.vstack([np.log10(n_list), np.ones(len(n_list))]).T, b=np.log10(dma_list), rcond=None)[0]
    return H

def hurst_dsod(x):
    if False:
        print('Hello World!')
    'Estimate Hurst exponent on data timeseries.\n\n    The estimation is based on the discrete second order derivative. Consists on\n    get two different noise of the original series and calculate the standard\n    deviation and calculate the slope of two point with that values.\n    source: https://gist.github.com/wmvanvliet/d883c3fe1402c7ced6fc\n\n    Parameters\n    ----------\n    x : numpy array\n        time series to estimate the Hurst exponent for.\n\n    Returns\n    -------\n    h : float\n        The estimation of the Hurst exponent for the given time series.\n\n    References\n    ----------\n    Istas, J.; G. Lang (1994), “Quadratic variations and estimation of the local\n    Hölder index of data Gaussian process,” Ann. Inst. Poincaré, 33, pp. 407–436.\n\n\n    Notes\n    -----\n    This hurst_ets is data literal traduction of wfbmesti.m of waveleet toolbox\n    from matlab.\n    '
    y = np.cumsum(np.diff(x, axis=0), axis=0)
    b1 = [1, -2, 1]
    y1 = signal.lfilter(b1, 1, y, axis=0)
    y1 = y1[len(b1) - 1:]
    b2 = [1, 0, -2, 0, 1]
    y2 = signal.lfilter(b2, 1, y, axis=0)
    y2 = y2[len(b2) - 1:]
    s1 = np.mean(y1 ** 2, axis=0)
    s2 = np.mean(y2 ** 2, axis=0)
    return 0.5 * np.log2(s2 / s1)