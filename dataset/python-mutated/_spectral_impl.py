"""
Spectral analysis functions and utilities.

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
import warnings
import cupy
from cupy_backends.cuda.api import runtime
import cupyx.scipy.signal._signaltools as filtering
from cupyx.scipy.signal._arraytools import odd_ext, even_ext, zero_ext, const_ext, _as_strided
from cupyx.scipy.signal.windows._windows import get_window

def _get_raw_typename(dtype):
    if False:
        for i in range(10):
            print('nop')
    return cupy.dtype(dtype).name

def _get_module_func_raw(module, func_name, *template_args):
    if False:
        return 10
    args_dtypes = [_get_raw_typename(arg.dtype) for arg in template_args]
    template = '_'.join(args_dtypes)
    kernel_name = f'{func_name}_{template}' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel
if runtime.is_hip:
    KERNEL_BASE = '\n    #include <hip/hip_runtime.h>\n'
else:
    KERNEL_BASE = '\n#include <cuda_runtime.h>\n#include <device_launch_parameters.h>\n'
LOMBSCARGLE_KERNEL = KERNEL_BASE + '\n\n///////////////////////////////////////////////////////////////////////////////\n//                            LOMBSCARGLE                                    //\n///////////////////////////////////////////////////////////////////////////////\n\ntemplate<typename T>\n__device__ void _cupy_lombscargle_float( const int x_shape,\n                                         const int freqs_shape,\n                                         const T *__restrict__ x,\n                                         const T *__restrict__ y,\n                                         const T *__restrict__ freqs,\n                                         T *__restrict__ pgram,\n                                         const T *__restrict__ y_dot ) {\n\n    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };\n    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };\n\n    T yD {};\n    if ( y_dot[0] == 0 ) {\n        yD = 1.0f;\n    } else {\n        yD = 2.0f / y_dot[0];\n    }\n\n    for ( int tid = tx; tid < freqs_shape; tid += stride ) {\n\n        T freq { freqs[tid] };\n\n        T xc {};\n        T xs {};\n        T cc {};\n        T ss {};\n        T cs {};\n        T c {};\n        T s {};\n\n        for ( int j = 0; j < x_shape; j++ ) {\n            sincosf( freq * x[j], &s, &c );\n            xc += y[j] * c;\n            xs += y[j] * s;\n            cc += c * c;\n            ss += s * s;\n            cs += c * s;\n        }\n\n        T c_tau {};\n        T s_tau {};\n        T tau { atan2f( 2.0f * cs, cc - ss ) / ( 2.0f * freq ) };\n        sincosf( freq * tau, &s_tau, &c_tau );\n        T c_tau2 { c_tau * c_tau };\n        T s_tau2 { s_tau * s_tau };\n        T cs_tau { 2.0f * c_tau * s_tau };\n\n        pgram[tid] = ( 0.5f * ( ( ( c_tau * xc + s_tau * xs ) *\n                                  ( c_tau * xc + s_tau * xs ) /\n                                  ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +\n                                ( ( c_tau * xs - s_tau * xc ) *\n                                  ( c_tau * xs - s_tau * xc ) /\n                                  ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) *\n                     yD;\n    }\n}\n\nextern "C" __global__ void __launch_bounds__( 512 ) _cupy_lombscargle_float32(\n        const int x_shape, const int freqs_shape, const float *__restrict__ x,\n        const float *__restrict__ y, const float *__restrict__ freqs,\n        float *__restrict__ pgram, const float *__restrict__ y_dot ) {\n    _cupy_lombscargle_float<float>( x_shape, freqs_shape, x, y,\n                                    freqs, pgram, y_dot );\n}\n\ntemplate<typename T>\n__device__ void _cupy_lombscargle_double( const int x_shape,\n                                          const int freqs_shape,\n                                          const T *__restrict__ x,\n                                          const T *__restrict__ y,\n                                          const T *__restrict__ freqs,\n                                          T *__restrict__ pgram,\n                                          const T *__restrict__ y_dot ) {\n\n    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };\n    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };\n\n    T yD {};\n    if ( y_dot[0] == 0 ) {\n        yD = 1.0;\n    } else {\n        yD = 2.0 / y_dot[0];\n    }\n\n    for ( int tid = tx; tid < freqs_shape; tid += stride ) {\n\n        T freq { freqs[tid] };\n\n        T xc {};\n        T xs {};\n        T cc {};\n        T ss {};\n        T cs {};\n        T c {};\n        T s {};\n\n        for ( int j = 0; j < x_shape; j++ ) {\n\n            sincos( freq * x[j], &s, &c );\n            xc += y[j] * c;\n            xs += y[j] * s;\n            cc += c * c;\n            ss += s * s;\n            cs += c * s;\n        }\n\n        T c_tau {};\n        T s_tau {};\n        T tau { atan2( 2.0 * cs, cc - ss ) / ( 2.0 * freq ) };\n        sincos( freq * tau, &s_tau, &c_tau );\n        T c_tau2 { c_tau * c_tau };\n        T s_tau2 { s_tau * s_tau };\n        T cs_tau { 2.0 * c_tau * s_tau };\n\n        pgram[tid] = ( 0.5 * ( ( ( c_tau * xc + s_tau * xs ) *\n                                 ( c_tau * xc + s_tau * xs ) /\n                                 ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +\n                               ( ( c_tau * xs - s_tau * xc ) *\n                                 ( c_tau * xs - s_tau * xc ) /\n                                 ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) *\n                     yD;\n    }\n}\n\nextern "C" __global__ void __launch_bounds__( 512 ) _cupy_lombscargle_float64(\n        const int x_shape, const int freqs_shape, const double *__restrict__ x,\n        const double *__restrict__ y, const double *__restrict__ freqs,\n        double *__restrict__ pgram, const double *__restrict__ y_dot ) {\n\n    _cupy_lombscargle_double<double>( x_shape, freqs_shape, x, y, freqs,\n                                      pgram, y_dot );\n}\n'
LOMBSCARGLE_MODULE = cupy.RawModule(code=LOMBSCARGLE_KERNEL, options=('-std=c++11',), name_expressions=['_cupy_lombscargle_float32', '_cupy_lombscargle_float64'])

def _lombscargle(x, y, freqs, pgram, y_dot):
    if False:
        while True:
            i = 10
    device_id = cupy.cuda.Device()
    num_blocks = device_id.attributes['MultiProcessorCount'] * 20
    block_sz = 512
    lombscargle_kernel = _get_module_func_raw(LOMBSCARGLE_MODULE, '_cupy_lombscargle', x)
    args = (x.shape[0], freqs.shape[0], x, y, freqs, pgram, y_dot)
    lombscargle_kernel((num_blocks,), (block_sz,), args)

def _spectral_helper(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd', boundary=None, padded=False):
    if False:
        while True:
            i = 10
    "\n    Calculate various forms of windowed FFTs for PSD, CSD, etc.\n\n    This is a helper function that implements the commonality between\n    the stft, psd, csd, and spectrogram functions. It is not designed to\n    be called externally. The windows are not averaged over; the result\n    from each window is returned.\n\n    Parameters\n    ---------\n    x : array_like\n        Array or sequence containing the data to be analyzed.\n    y : array_like\n        Array or sequence containing the data to be analyzed. If this is\n        the same object in memory as `x` (i.e. ``_spectral_helper(x,\n        x, ...)``), the extra computations are spared.\n    fs : float, optional\n        Sampling frequency of the time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to 'constant'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Defaults to `True`, but for\n        complex data, a two-sided spectrum is always returned.\n    scaling : { 'density', 'spectrum' }, optional\n        Selects between computing the cross spectral density ('density')\n        where `Pxy` has units of V**2/Hz and computing the cross\n        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`\n        and `y` are measured in V and `fs` is measured in Hz.\n        Defaults to 'density'\n    axis : int, optional\n        Axis along which the FFTs are computed; the default is over the\n        last axis (i.e. ``axis=-1``).\n    mode: str {'psd', 'stft'}, optional\n        Defines what kind of return values are expected. Defaults to\n        'psd'.\n    boundary : str or None, optional\n        Specifies whether the input signal is extended at both ends, and\n        how to generate the new values, in order to center the first\n        windowed segment on the first input point. This has the benefit\n        of enabling reconstruction of the first input point when the\n        employed window function starts at zero. Valid options are\n        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to\n        `None`.\n    padded : bool, optional\n        Specifies whether the input signal is zero-padded at the end to\n        make the signal fit exactly into an integer number of window\n        segments, so that all of the signal is included in the output.\n        Defaults to `False`. Padding occurs after boundary extension, if\n        `boundary` is not `None`, and `padded` is `True`.\n\n    Returns\n    -------\n    freqs : ndarray\n        Array of sample frequencies.\n    t : ndarray\n        Array of times corresponding to each data segment\n    result : ndarray\n        Array of output data, contents dependent on *mode* kwarg.\n\n    Notes\n    -----\n    Adapted from matplotlib.mlab\n\n    "
    if mode not in ['psd', 'stft']:
        raise ValueError(f"Unknown value for mode {mode}, must be one of: {{'psd', 'stft'}}")
    boundary_funcs = {'even': even_ext, 'odd': odd_ext, 'constant': const_ext, 'zeros': zero_ext, None: None}
    if boundary not in boundary_funcs:
        raise ValueError("Unknown boundary option '{0}', must be one of: {1}".format(boundary, list(boundary_funcs.keys())))
    same_data = y is x
    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is 'stft'")
    axis = int(axis)
    x = cupy.asarray(x)
    if not same_data:
        y = cupy.asarray(y)
        outdtype = cupy.result_type(x, y, cupy.complex64)
    else:
        outdtype = cupy.result_type(x, cupy.complex64)
    if not same_data:
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = cupy.broadcast(cupy.empty(xouter), cupy.empty(youter)).shape
        except ValueError:
            raise ValueError('x and y cannot be broadcast together.')
    if same_data:
        if x.size == 0:
            return (cupy.empty(x.shape), cupy.empty(x.shape), cupy.empty(x.shape))
    elif x.size == 0 or y.size == 0:
        outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
        emptyout = cupy.rollaxis(cupy.empty(outshape), -1, axis)
        return (emptyout, emptyout, emptyout)
    if x.ndim > 1:
        if axis != -1:
            x = cupy.rollaxis(x, axis, len(x.shape))
            if not same_data and y.ndim > 1:
                y = cupy.rollaxis(y, axis, len(y.shape))
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = cupy.concatenate((x, cupy.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = cupy.concatenate((y, cupy.zeros(pad_shape)), -1)
    if nperseg is not None:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')
    (win, nperseg) = _triage_segments(window, nperseg, input_length=x.shape[-1])
    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)
    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap
    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg // 2, axis=-1)
        if not same_data:
            y = ext_func(y, nperseg // 2, axis=-1)
    if padded:
        nadd = -(x.shape[-1] - nperseg) % nstep % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = cupy.concatenate((x, cupy.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = cupy.concatenate((y, cupy.zeros(zeros_shape)), axis=-1)
    if not detrend:

        def detrend_func(d):
            if False:
                i = 10
                return i + 15
            return d
    elif not hasattr(detrend, '__call__'):

        def detrend_func(d):
            if False:
                return 10
            return filtering.detrend(d, type=detrend, axis=-1)
    elif axis != -1:

        def detrend_func(d):
            if False:
                for i in range(10):
                    print('nop')
            d = cupy.rollaxis(d, -1, axis)
            d = detrend(d)
            return cupy.rollaxis(d, axis, len(d.shape))
    else:
        detrend_func = detrend
    if cupy.result_type(win, cupy.complex64) != outdtype:
        win = win.astype(outdtype)
    if scaling == 'density':
        scale = 1.0 / (fs * (win * win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum() ** 2
    else:
        raise ValueError('Unknown scaling: %r' % scaling)
    if mode == 'stft':
        scale = cupy.sqrt(scale)
    if return_onesided:
        if cupy.iscomplexobj(x):
            sides = 'twosided'
            warnings.warn('Input data is complex, switching to return_onesided=False')
        else:
            sides = 'onesided'
            if not same_data:
                if cupy.iscomplexobj(y):
                    sides = 'twosided'
                    warnings.warn('Input data is complex, switching to return_onesided=False')
    else:
        sides = 'twosided'
    if sides == 'twosided':
        freqs = cupy.fft.fftfreq(nfft, 1 / fs)
    elif sides == 'onesided':
        freqs = cupy.fft.rfftfreq(nfft, 1 / fs)
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)
    if not same_data:
        result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft, sides)
        result = cupy.conj(result) * result_y
    elif mode == 'psd':
        result = cupy.conj(result) * result
    result *= scale
    if sides == 'onesided' and mode == 'psd':
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            result[..., 1:-1] *= 2
    time = cupy.arange(nperseg / 2, x.shape[-1] - nperseg / 2 + 1, nperseg - noverlap) / float(fs)
    if boundary is not None:
        time -= nperseg / 2 / fs
    result = result.astype(outdtype)
    if same_data and mode != 'stft':
        result = result.real
    if axis < 0:
        axis -= 1
    result = cupy.rollaxis(result, -1, axis)
    return (freqs, time, result)

def _triage_segments(window, nperseg, input_length):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parses window and nperseg arguments for spectrogram and _spectral_helper.\n    This is a helper function, not meant to be called externally.\n\n    Parameters\n    ----------\n    window : string, tuple, or ndarray\n        If window is specified by a string or tuple and nperseg is not\n        specified, nperseg is set to the default of 256 and returns a window of\n        that length.\n        If instead the window is array_like and nperseg is not specified, then\n        nperseg is set to the length of the window. A ValueError is raised if\n        the user supplies both an array_like window and a value for nperseg but\n        nperseg does not equal the length of the window.\n\n    nperseg : int\n        Length of each segment\n\n    input_length: int\n        Length of input signal, i.e. x.shape[-1]. Used to test for errors.\n\n    Returns\n    -------\n    win : ndarray\n        window. If function was called with string or tuple than this will hold\n        the actual array used as a window.\n\n    nperseg : int\n        Length of each segment. If window is str or tuple, nperseg is set to\n        256. If window is array_like, nperseg is set to the length of the\n        6\n        window.\n    '
    if isinstance(window, str) or isinstance(window, tuple):
        if nperseg is None:
            nperseg = 256
        if nperseg > input_length:
            warnings.warn('nperseg = {0:d} is greater than input length  = {1:d}, using nperseg = {1:d}'.format(nperseg, input_length))
            nperseg = input_length
        win = get_window(window, nperseg)
    else:
        win = cupy.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if input_length < win.shape[-1]:
            raise ValueError('window is longer than input signal')
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError('value specified for nperseg is different from length of window')
    return (win, nperseg)

def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):
    if False:
        return 10
    '\n    Calculate windowed FFT, for internal use by\n    cusignal.spectral_analysis.spectral._spectral_helper\n\n    This is a helper function that does the main FFT calculation for\n    `_spectral helper`. All input validation is performed there, and the\n    data axis is assumed to be the last axis of x. It is not designed to\n    be called externally. The windows are not averaged over; the result\n    from each window is returned.\n\n    Returns\n    -------\n    result : ndarray\n        Array of FFT data\n\n    Notes\n    -----\n    Adapted from matplotlib.mlab\n\n    '
    if nperseg == 1 and noverlap == 0:
        result = x[..., cupy.newaxis]
    else:
        step = nperseg - noverlap
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        result = _as_strided(x, shape=shape, strides=strides)
    result = detrend_func(result)
    result = win * result
    if sides == 'twosided':
        func = cupy.fft.fft
    else:
        result = result.real
        func = cupy.fft.rfft
    result = func(result, n=nfft)
    return result

def _median_bias(n):
    if False:
        return 10
    '\n    Returns the bias of the median of a set of periodograms relative to\n    the mean.\n\n    See arXiv:gr-qc/0509116 Appendix B for details.\n\n    Parameters\n    ----------\n    n : int\n        Numbers of periodograms being averaged.\n\n    Returns\n    -------\n    bias : float\n        Calculated bias.\n    '
    ii_2 = 2 * cupy.arange(1.0, (n - 1) // 2 + 1)
    return 1 + cupy.sum(1.0 / (ii_2 + 1) - 1.0 / ii_2)