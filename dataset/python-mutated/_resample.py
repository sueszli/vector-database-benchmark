"""
Signal sampling functions.

Some of the functions defined here were ported directly from CuSignal under
terms of the MIT license, under the following notice:

Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import operator
from math import gcd
import cupy
from cupyx.scipy.fft import fft, rfft, fftfreq, ifft, irfft, ifftshift
from cupyx.scipy.signal._iir_filter_design import cheby1
from cupyx.scipy.signal._fir_filter_design import firwin
from cupyx.scipy.signal._iir_filter_conversions import zpk2sos
from cupyx.scipy.signal._ltisys import dlti
from cupyx.scipy.signal._upfirdn import upfirdn, _output_len
from cupyx.scipy.signal._signaltools import sosfiltfilt, filtfilt, sosfilt, lfilter
from cupyx.scipy.signal.windows._windows import get_window

def _design_resample_poly(up, down, window):
    if False:
        print('Hello World!')
    '\n    Design a prototype FIR low-pass filter using the window method\n    for use in polyphase rational resampling.\n\n    Parameters\n    ----------\n    up : int\n        The upsampling factor.\n    down : int\n        The downsampling factor.\n    window : string or tuple\n        Desired window to use to design the low-pass filter.\n        See below for details.\n\n    Returns\n    -------\n    h : array\n        The computed FIR filter coefficients.\n\n    See Also\n    --------\n    resample_poly : Resample up or down using the polyphase method.\n\n    Notes\n    -----\n    The argument `window` specifies the FIR low-pass filter design.\n    The functions `cusignal.get_window` and `cusignal.firwin`\n    are called to generate the appropriate filter coefficients.\n\n    The returned array of coefficients will always be of data type\n    `complex128` to maintain precision. For use in lower-precision\n    filter operations, this array should be converted to the desired\n    data type before providing it to `cusignal.resample_poly`.\n\n    '
    g_ = gcd(up, down)
    up //= g_
    down //= g_
    max_rate = max(up, down)
    f_c = 1.0 / max_rate
    half_len = 10 * max_rate
    h = firwin(2 * half_len + 1, f_c, window=window)
    return h

def decimate(x, q, n=None, ftype='iir', axis=-1, zero_phase=True):
    if False:
        print('Hello World!')
    "\n    Downsample the signal after applying an anti-aliasing filter.\n\n    By default, an order 8 Chebyshev type I filter is used. A 30 point FIR\n    filter with Hamming window is used if `ftype` is 'fir'.\n\n    Parameters\n    ----------\n    x : array_like\n        The signal to be downsampled, as an N-dimensional array.\n    q : int\n        The downsampling factor. When using IIR downsampling, it is recommended\n        to call `decimate` multiple times for downsampling factors higher than\n        13.\n    n : int, optional\n        The order of the filter (1 less than the length for 'fir'). Defaults to\n        8 for 'iir' and 20 times the downsampling factor for 'fir'.\n    ftype : str {'iir', 'fir'} or ``dlti`` instance, optional\n        If 'iir' or 'fir', specifies the type of lowpass filter. If an instance\n        of an `dlti` object, uses that object to filter before downsampling.\n    axis : int, optional\n        The axis along which to decimate.\n    zero_phase : bool, optional\n        Prevent phase shift by filtering with `filtfilt` instead of `lfilter`\n        when using an IIR filter, and shifting the outputs back by the filter's\n        group delay when using an FIR filter. The default value of ``True`` is\n        recommended, since a phase shift is generally not desired.\n\n    Returns\n    -------\n    y : ndarray\n        The down-sampled signal.\n\n    See Also\n    --------\n    resample : Resample up or down using the FFT method.\n    resample_poly : Resample using polyphase filtering and an FIR filter.\n    "
    x = cupy.asarray(x)
    q = operator.index(q)
    if n is not None:
        n = operator.index(n)
    result_type = x.dtype
    if not cupy.issubdtype(result_type, cupy.inexact) or result_type.type == cupy.float16:
        result_type = cupy.float64
    if ftype == 'fir':
        if n is None:
            half_len = 10 * q
            n = 2 * half_len
        (b, a) = (firwin(n + 1, 1.0 / q, window='hamming'), 1.0)
        b = cupy.asarray(b, dtype=result_type)
        a = cupy.asarray(a, dtype=result_type)
    elif ftype == 'iir':
        iir_use_sos = True
        if n is None:
            n = 8
        sos = cheby1(n, 0.05, 0.8 / q, output='sos')
        sos = cupy.asarray(sos, dtype=result_type)
    elif isinstance(ftype, dlti):
        system = ftype._as_zpk()
        if system.poles.shape[0] == 0:
            system = ftype._as_tf()
            (b, a) = (system.num, system.den)
            ftype = 'fir'
        elif any(cupy.iscomplex(system.poles)) or any(cupy.iscomplex(system.poles)) or cupy.iscomplex(system.gain):
            iir_use_sos = False
            system = ftype._as_tf()
            (b, a) = (system.num, system.den)
        else:
            iir_use_sos = True
            sos = zpk2sos(system.zeros, system.poles, system.gain)
            sos = cupy.asarray(sos, dtype=result_type)
    else:
        raise ValueError('invalid ftype')
    sl = [slice(None)] * x.ndim
    if ftype == 'fir':
        b = b / a
        if zero_phase:
            y = resample_poly(x, 1, q, axis=axis, window=b)
        else:
            n_out = x.shape[axis] // q + bool(x.shape[axis] % q)
            y = upfirdn(b, x, up=1, down=q, axis=axis)
            sl[axis] = slice(None, n_out, None)
    else:
        if zero_phase:
            if iir_use_sos:
                y = sosfiltfilt(sos, x, axis=axis)
            else:
                y = filtfilt(b, a, x, axis=axis)
        elif iir_use_sos:
            y = sosfilt(sos, x, axis=axis)
        else:
            y = lfilter(b, a, x, axis=axis)
        sl[axis] = slice(None, None, q)
    return y[tuple(sl)]

def resample(x, num, t=None, axis=0, window=None, domain='time'):
    if False:
        print('Hello World!')
    "\n    Resample `x` to `num` samples using Fourier method along the given axis.\n\n    The resampled signal starts at the same value as `x` but is sampled\n    with a spacing of ``len(x) / num * (spacing of x)``.  Because a\n    Fourier method is used, the signal is assumed to be periodic.\n\n    Parameters\n    ----------\n    x : array_like\n        The data to be resampled.\n    num : int\n        The number of samples in the resampled signal.\n    t : array_like, optional\n        If `t` is given, it is assumed to be the sample positions\n        associated with the signal data in `x`.\n    axis : int, optional\n        The axis of `x` that is resampled.  Default is 0.\n    window : array_like, callable, string, float, or tuple, optional\n        Specifies the window applied to the signal in the Fourier\n        domain.  See below for details.\n    domain : string, optional\n        A string indicating the domain of the input `x`:\n\n        ``time``\n           Consider the input `x` as time-domain. (Default)\n        ``freq``\n           Consider the input `x` as frequency-domain.\n\n    Returns\n    -------\n    resampled_x or (resampled_x, resampled_t)\n        Either the resampled array, or, if `t` was given, a tuple\n        containing the resampled array and the corresponding resampled\n        positions.\n\n    See Also\n    --------\n    decimate : Downsample the signal after applying an FIR or IIR filter.\n    resample_poly : Resample using polyphase filtering and an FIR filter.\n\n    Notes\n    -----\n    The argument `window` controls a Fourier-domain window that tapers\n    the Fourier spectrum before zero-padding to alleviate ringing in\n    the resampled values for sampled signals you didn't intend to be\n    interpreted as band-limited.\n\n    If `window` is a function, then it is called with a vector of inputs\n    indicating the frequency bins (i.e. fftfreq(x.shape[axis]) ).\n\n    If `window` is an array of the same length as `x.shape[axis]` it is\n    assumed to be the window to be applied directly in the Fourier\n    domain (with dc and low-frequency first).\n\n    For any other type of `window`, the function `cusignal.get_window`\n    is called to generate the window.\n\n    The first sample of the returned vector is the same as the first\n    sample of the input vector.  The spacing between samples is changed\n    from ``dx`` to ``dx * len(x) / num``.\n\n    If `t` is not None, then it represents the old sample positions,\n    and the new sample positions will be returned as well as the new\n    samples.\n\n    As noted, `resample` uses FFT transformations, which can be very\n    slow if the number of input or output samples is large and prime;\n    see `scipy.fftpack.fft`.\n\n    Examples\n    --------\n    Note that the end of the resampled data rises to meet the first\n    sample of the next cycle:\n\n    >>> import cupy as cp\n    >>> import cupyx.scipy.signal import resample\n\n    >>> x = cupy.linspace(0, 10, 20, endpoint=False)\n    >>> y = cupy.cos(-x**2/6.0)\n    >>> f = resample(y, 100)\n    >>> xnew = cupy.linspace(0, 10, 100, endpoint=False)\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(cupy.asnumpy(x), cupy.asnumpy(y), 'go-', cupy.asnumpy(xnew),                 cupy.asnumpy(f), '.-', 10, cupy.asnumpy(y[0]), 'ro')\n    >>> plt.legend(['data', 'resampled'], loc='best')\n    >>> plt.show()\n    "
    if domain not in ('time', 'freq'):
        raise ValueError("Acceptable domain flags are 'time' or 'freq', not domain={}".format(domain))
    x = cupy.asarray(x)
    Nx = x.shape[axis]
    real_input = cupy.isrealobj(x)
    if domain == 'time':
        if real_input:
            X = rfft(x, axis=axis)
        else:
            X = fft(x, axis=axis)
    else:
        X = x
    if window is not None:
        if callable(window):
            W = window(fftfreq(Nx))
        elif isinstance(window, cupy.ndarray):
            if window.shape != (Nx,):
                raise ValueError('window must have the same length as data')
            W = window
        else:
            W = ifftshift(get_window(window, Nx))
        newshape_W = [1] * x.ndim
        newshape_W[axis] = X.shape[axis]
        if real_input:
            W_real = W.copy()
            W_real[1:] += W_real[-1:0:-1]
            W_real[1:] *= 0.5
            X *= W_real[:newshape_W[axis]].reshape(newshape_W)
        else:
            X *= W.reshape(newshape_W)
    newshape = list(x.shape)
    if real_input:
        newshape[axis] = num // 2 + 1
    else:
        newshape[axis] = num
    Y = cupy.zeros(newshape, X.dtype)
    N = min(num, Nx)
    nyq = N // 2 + 1
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]
    if not real_input:
        if N > 2:
            sl[axis] = slice(nyq - N, None)
            Y[tuple(sl)] = X[tuple(sl)]
    if N % 2 == 0:
        if num < Nx:
            if real_input:
                sl[axis] = slice(N // 2, N // 2 + 1)
                Y[tuple(sl)] *= 2.0
            else:
                sl[axis] = slice(-N // 2, -N // 2 + 1)
                Y[tuple(sl)] += X[tuple(sl)]
        elif Nx < num:
            sl[axis] = slice(N // 2, N // 2 + 1)
            Y[tuple(sl)] *= 0.5
            if not real_input:
                temp = Y[tuple(sl)]
                sl[axis] = slice(num - N // 2, num - N // 2 + 1)
                Y[tuple(sl)] = temp
    if real_input:
        y = irfft(Y, num, axis=axis)
    else:
        y = ifft(Y, axis=axis, overwrite_x=True)
    y *= float(num) / float(Nx)
    if t is None:
        return y
    else:
        new_t = cupy.arange(0, num) * (t[1] - t[0]) * Nx / float(num) + t[0]
        return (y, new_t)

def resample_poly(x, up, down, axis=0, window=('kaiser', 5.0), padtype='constant', cval=None):
    if False:
        while True:
            i = 10
    "\n    Resample `x` along the given axis using polyphase filtering.\n\n    The signal `x` is upsampled by the factor `up`, a zero-phase low-pass\n    FIR filter is applied, and then it is downsampled by the factor `down`.\n    The resulting sample rate is ``up / down`` times the original sample\n    rate. Values beyond the boundary of the signal are assumed to be zero\n    during the filtering step.\n\n    Parameters\n    ----------\n    x : array_like\n        The data to be resampled.\n    up : int\n        The upsampling factor.\n    down : int\n        The downsampling factor.\n    axis : int, optional\n        The axis of `x` that is resampled. Default is 0.\n    window : string, tuple, or array_like, optional\n        Desired window to use to design the low-pass filter, or the FIR filter\n        coefficients to employ. See below for details.\n    padtype : string, optional\n        `constant`, `line`, `mean`, `median`, `maximum`, `minimum` or any of\n        the other signal extension modes supported by\n        `cupyx.scipy.signal.upfirdn`. Changes assumptions on values beyond\n        the boundary. If `constant`, assumed to be `cval` (default zero).\n        If `line` assumed to continue a linear trend defined by the first and\n        last points. `mean`, `median`, `maximum` and `minimum` work as in\n        `cupy.pad` and assume that the values beyond the boundary are the mean,\n        median, maximum or minimum respectively of the array along the axis.\n    cval : float, optional\n        Value to use if `padtype='constant'`. Default is zero.\n\n    Returns\n    -------\n    resampled_x : array\n        The resampled array.\n\n    See Also\n    --------\n    decimate : Downsample the signal after applying an FIR or IIR filter.\n    resample : Resample up or down using the FFT method.\n\n    Notes\n    -----\n    This polyphase method will likely be faster than the Fourier method\n    in `cusignal.resample` when the number of samples is large and\n    prime, or when the number of samples is large and `up` and `down`\n    share a large greatest common denominator. The length of the FIR\n    filter used will depend on ``max(up, down) // gcd(up, down)``, and\n    the number of operations during polyphase filtering will depend on\n    the filter length and `down` (see `cusignal.upfirdn` for details).\n\n    The argument `window` specifies the FIR low-pass filter design.\n\n    If `window` is an array_like it is assumed to be the FIR filter\n    coefficients. Note that the FIR filter is applied after the upsampling\n    step, so it should be designed to operate on a signal at a sampling\n    frequency higher than the original by a factor of `up//gcd(up, down)`.\n    This function's output will be centered with respect to this array, so it\n    is best to pass a symmetric filter with an odd number of samples if, as\n    is usually the case, a zero-phase filter is desired.\n\n    For any other type of `window`, the functions `cusignal.get_window`\n    and `cusignal.firwin` are called to generate the appropriate filter\n    coefficients.\n\n    The first sample of the returned vector is the same as the first\n    sample of the input vector. The spacing between samples is changed\n    from ``dx`` to ``dx * down / float(up)``.\n\n    Examples\n    --------\n    Note that the end of the resampled data rises to meet the first\n    sample of the next cycle for the FFT method, and gets closer to zero\n    for the polyphase method:\n\n    >>> import cupy\n    >>> import cupyx.scipy.signal import resample, resample_poly\n\n    >>> x = cupy.linspace(0, 10, 20, endpoint=False)\n    >>> y = cupy.cos(-x**2/6.0)\n    >>> f_fft = resample(y, 100)\n    >>> f_poly = resample_poly(y, 100, 20)\n    >>> xnew = cupy.linspace(0, 10, 100, endpoint=False)\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(cupy.asnumpy(xnew), cupy.asnumpy(f_fft), 'b.-',                  cupy.asnumpy(xnew), cupy.asnumpy(f_poly), 'r.-')\n    >>> plt.plot(cupy.asnumpy(x), cupy.asnumpy(y), 'ko-')\n    >>> plt.plot(10, cupy.asnumpy(y[0]), 'bo', 10, 0., 'ro')  # boundaries\n    >>> plt.legend(['resample', 'resamp_poly', 'data'], loc='best')\n    >>> plt.show()\n    "
    if padtype != 'constant' or cval is not None:
        raise ValueError('padtype and cval arguments are not supported by upfirdn')
    x = cupy.asarray(x)
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError('up and down must be >= 1')
    g_ = gcd(up, down)
    up //= g_
    down //= g_
    if up == down == 1:
        return x.copy()
    n_out = x.shape[axis] * up
    n_out = n_out // down + bool(n_out % down)
    if isinstance(window, (list, cupy.ndarray)):
        window = cupy.asarray(window)
        if window.ndim > 1:
            raise ValueError('window must be 1-D')
        half_len = (window.size - 1) // 2
        h = up * window
    else:
        half_len = 10 * max(up, down)
        h = up * _design_resample_poly(up, down, window)
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    while _output_len(len(h) + n_pre_pad + n_post_pad, x.shape[axis], up, down) < n_out + n_pre_remove:
        n_post_pad += 1
    h = cupy.concatenate((cupy.zeros(n_pre_pad, h.dtype), h, cupy.zeros(n_post_pad, h.dtype)))
    n_pre_remove_end = n_pre_remove + n_out
    y = upfirdn(h, x, up, down, axis)
    keep = [slice(None)] * x.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)
    return y[tuple(keep)]