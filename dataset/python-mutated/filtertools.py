"""Linear Filters for time series analysis and testing


TODO:
* check common sequence in signature of filter functions (ar,ma,x) or (x,ar,ma)

Created on Sat Oct 23 17:18:03 2010

Author: Josef-pktd
"""
import numpy as np
import scipy.fftpack as fft
from scipy import signal
try:
    from scipy.signal._signaltools import _centered as trim_centered
except ImportError:
    from scipy.signal.signaltools import _centered as trim_centered
from statsmodels.tools.validation import array_like, PandasWrapper

def _pad_nans(x, head=None, tail=None):
    if False:
        while True:
            i = 10
    if np.ndim(x) == 1:
        if head is None and tail is None:
            return x
        elif head and tail:
            return np.r_[[np.nan] * head, x, [np.nan] * tail]
        elif tail is None:
            return np.r_[[np.nan] * head, x]
        elif head is None:
            return np.r_[x, [np.nan] * tail]
    elif np.ndim(x) == 2:
        if head is None and tail is None:
            return x
        elif head and tail:
            return np.r_[[[np.nan] * x.shape[1]] * head, x, [[np.nan] * x.shape[1]] * tail]
        elif tail is None:
            return np.r_[[[np.nan] * x.shape[1]] * head, x]
        elif head is None:
            return np.r_[x, [[np.nan] * x.shape[1]] * tail]
    else:
        raise ValueError('Nan-padding for ndim > 2 not implemented')

def fftconvolveinv(in1, in2, mode='full'):
    if False:
        i = 10
        return i + 15
    '\n    Convolve two N-dimensional arrays using FFT. See convolve.\n\n    copied from scipy.signal.signaltools, but here used to try out inverse\n    filter. does not work or I cannot get it to work\n\n    2010-10-23:\n    looks ok to me for 1d,\n    from results below with padded data array (fftp)\n    but it does not work for multidimensional inverse filter (fftn)\n    original signal.fftconvolve also uses fftn\n    '
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = np.issubdtype(in1.dtype, np.complex) or np.issubdtype(in2.dtype, np.complex)
    size = s1 + s2 - 1
    fsize = 2 ** np.ceil(np.log2(size))
    IN1 = fft.fftn(in1, fsize)
    IN1 /= fft.fftn(in2, fsize)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = fft.ifftn(IN1)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == 'full':
        return ret
    elif mode == 'same':
        if np.product(s1, axis=0) > np.product(s2, axis=0):
            osize = s1
        else:
            osize = s2
        return trim_centered(ret, osize)
    elif mode == 'valid':
        return trim_centered(ret, abs(s2 - s1) + 1)

def fftconvolve3(in1, in2=None, in3=None, mode='full'):
    if False:
        print('Hello World!')
    "\n    Convolve two N-dimensional arrays using FFT. See convolve.\n\n    For use with arma  (old version: in1=num in2=den in3=data\n\n    * better for consistency with other functions in1=data in2=num in3=den\n    * note in2 and in3 need to have consistent dimension/shape\n      since I'm using max of in2, in3 shapes and not the sum\n\n    copied from scipy.signal.signaltools, but here used to try out inverse\n    filter does not work or I cannot get it to work\n\n    2010-10-23\n    looks ok to me for 1d,\n    from results below with padded data array (fftp)\n    but it does not work for multidimensional inverse filter (fftn)\n    original signal.fftconvolve also uses fftn\n    "
    if in2 is None and in3 is None:
        raise ValueError('at least one of in2 and in3 needs to be given')
    s1 = np.array(in1.shape)
    if in2 is not None:
        s2 = np.array(in2.shape)
    else:
        s2 = 0
    if in3 is not None:
        s3 = np.array(in3.shape)
        s2 = max(s2, s3)
    complex_result = np.issubdtype(in1.dtype, np.complex) or np.issubdtype(in2.dtype, np.complex)
    size = s1 + s2 - 1
    fsize = 2 ** np.ceil(np.log2(size))
    IN1 = in1.copy()
    if in2 is not None:
        IN1 = fft.fftn(in2, fsize)
    if in3 is not None:
        IN1 /= fft.fftn(in3, fsize)
    IN1 *= fft.fftn(in1, fsize)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = fft.ifftn(IN1)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == 'full':
        return ret
    elif mode == 'same':
        if np.product(s1, axis=0) > np.product(s2, axis=0):
            osize = s1
        else:
            osize = s2
        return trim_centered(ret, osize)
    elif mode == 'valid':
        return trim_centered(ret, abs(s2 - s1) + 1)

def recursive_filter(x, ar_coeff, init=None):
    if False:
        while True:
            i = 10
    '\n    Autoregressive, or recursive, filtering.\n\n    Parameters\n    ----------\n    x : array_like\n        Time-series data. Should be 1d or n x 1.\n    ar_coeff : array_like\n        AR coefficients in reverse time order. See Notes for details.\n    init : array_like\n        Initial values of the time-series prior to the first value of y.\n        The default is zero.\n\n    Returns\n    -------\n    array_like\n        Filtered array, number of columns determined by x and ar_coeff. If x\n        is a pandas object than a Series is returned.\n\n    Notes\n    -----\n    Computes the recursive filter ::\n\n        y[n] = ar_coeff[0] * y[n-1] + ...\n                + ar_coeff[n_coeff - 1] * y[n - n_coeff] + x[n]\n\n    where n_coeff = len(n_coeff).\n    '
    pw = PandasWrapper(x)
    x = array_like(x, 'x')
    ar_coeff = array_like(ar_coeff, 'ar_coeff')
    if init is not None:
        init = array_like(init, 'init')
        if len(init) != len(ar_coeff):
            raise ValueError('ar_coeff must be the same length as init')
    if init is not None:
        zi = signal.lfiltic([1], np.r_[1, -ar_coeff], init, x)
    else:
        zi = None
    y = signal.lfilter([1.0], np.r_[1, -ar_coeff], x, zi=zi)
    if init is not None:
        result = y[0]
    else:
        result = y
    return pw.wrap(result)

def convolution_filter(x, filt, nsides=2):
    if False:
        while True:
            i = 10
    '\n    Linear filtering via convolution. Centered and backward displaced moving\n    weighted average.\n\n    Parameters\n    ----------\n    x : array_like\n        data array, 1d or 2d, if 2d then observations in rows\n    filt : array_like\n        Linear filter coefficients in reverse time-order. Should have the\n        same number of dimensions as x though if 1d and ``x`` is 2d will be\n        coerced to 2d.\n    nsides : int, optional\n        If 2, a centered moving average is computed using the filter\n        coefficients. If 1, the filter coefficients are for past values only.\n        Both methods use scipy.signal.convolve.\n\n    Returns\n    -------\n    y : ndarray, 2d\n        Filtered array, number of columns determined by x and filt. If a\n        pandas object is given, a pandas object is returned. The index of\n        the return is the exact same as the time period in ``x``\n\n    Notes\n    -----\n    In nsides == 1, x is filtered ::\n\n        y[n] = filt[0]*x[n-1] + ... + filt[n_filt-1]*x[n-n_filt]\n\n    where n_filt is len(filt).\n\n    If nsides == 2, x is filtered around lag 0 ::\n\n        y[n] = filt[0]*x[n - n_filt/2] + ... + filt[n_filt / 2] * x[n]\n               + ... + x[n + n_filt/2]\n\n    where n_filt is len(filt). If n_filt is even, then more of the filter\n    is forward in time than backward.\n\n    If filt is 1d or (nlags,1) one lag polynomial is applied to all\n    variables (columns of x). If filt is 2d, (nlags, nvars) each series is\n    independently filtered with its own lag polynomial, uses loop over nvar.\n    This is different than the usual 2d vs 2d convolution.\n\n    Filtering is done with scipy.signal.convolve, so it will be reasonably\n    fast for medium sized data. For large data fft convolution would be\n    faster.\n    '
    if nsides == 1:
        trim_head = len(filt) - 1
        trim_tail = None
    elif nsides == 2:
        trim_head = int(np.ceil(len(filt) / 2.0) - 1) or None
        trim_tail = int(np.ceil(len(filt) / 2.0) - len(filt) % 2) or None
    else:
        raise ValueError('nsides must be 1 or 2')
    pw = PandasWrapper(x)
    x = array_like(x, 'x', maxdim=2)
    filt = array_like(filt, 'filt', ndim=x.ndim)
    if filt.ndim == 1 or min(filt.shape) == 1:
        result = signal.convolve(x, filt, mode='valid')
    else:
        nlags = filt.shape[0]
        nvar = x.shape[1]
        result = np.zeros((x.shape[0] - nlags + 1, nvar))
        if nsides == 2:
            for i in range(nvar):
                result[:, i] = signal.convolve(x[:, i], filt[:, i], mode='valid')
        elif nsides == 1:
            for i in range(nvar):
                result[:, i] = signal.convolve(x[:, i], np.r_[0, filt[:, i]], mode='valid')
    result = _pad_nans(result, trim_head, trim_tail)
    return pw.wrap(result)

def miso_lfilter(ar, ma, x, useic=False):
    if False:
        i = 10
        return i + 15
    '\n    Filter multiple time series into a single time series.\n\n    Uses a convolution to merge inputs, and then lfilter to produce output.\n\n    Parameters\n    ----------\n    ar : array_like\n        The coefficients of autoregressive lag polynomial including lag zero,\n        ar(L) in the expression ar(L)y_t.\n    ma : array_like, same ndim as x, currently 2d\n        The coefficient of the moving average lag polynomial, ma(L) in\n        ma(L)x_t.\n    x : array_like\n        The 2-d input data series, time in rows, variables in columns.\n    useic : bool\n        Flag indicating whether to use initial conditions.\n\n    Returns\n    -------\n    y : ndarray\n        The filtered output series.\n    inp : ndarray, 1d\n        The combined input series.\n\n    Notes\n    -----\n    currently for 2d inputs only, no choice of axis\n    Use of signal.lfilter requires that ar lag polynomial contains\n    floating point numbers\n    does not cut off invalid starting and final values\n\n    miso_lfilter find array y such that:\n\n            ar(L)y_t = ma(L)x_t\n\n    with shapes y (nobs,), x (nobs, nvars), ar (narlags,), and\n    ma (narlags, nvars).\n    '
    ma = array_like(ma, 'ma')
    ar = array_like(ar, 'ar')
    inp = signal.correlate(x, ma[::-1, :])[:, (x.shape[1] + 1) // 2]
    nobs = x.shape[0]
    if useic:
        return (signal.lfilter([1], ar, inp, zi=signal.lfiltic(np.array([1.0, 0.0]), ar, useic))[0][:nobs], inp[:nobs])
    else:
        return (signal.lfilter([1], ar, inp)[:nobs], inp[:nobs])