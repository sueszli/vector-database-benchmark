"""
Waveform-generating functions.

Some of the functions defined here were ported directly from CuSignal under
terms of the MIT license, under the following notice:

Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
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
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
import numpy as np

def _get_typename(dtype):
    if False:
        for i in range(10):
            print('nop')
    typename = get_typename(dtype)
    if cupy.dtype(dtype).kind == 'c':
        typename = 'thrust::' + typename
    elif typename == 'float16':
        if runtime.is_hip:
            typename = '__half'
        else:
            typename = 'half'
    return typename
FLOAT_TYPES = [cupy.float16, cupy.float32, cupy.float64]
INT_TYPES = [cupy.int8, cupy.int16, cupy.int32, cupy.int64]
UNSIGNED_TYPES = [cupy.uint8, cupy.uint16, cupy.uint32, cupy.uint64]
COMPLEX_TYPES = [cupy.complex64, cupy.complex128]
TYPES = FLOAT_TYPES + INT_TYPES + UNSIGNED_TYPES + COMPLEX_TYPES
TYPE_NAMES = [_get_typename(t) for t in TYPES]

def _get_module_func(module, func_name, *template_args):
    if False:
        print('Hello World!')
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel
_sawtooth_kernel = cupy.ElementwiseKernel('T t, T w', 'float64 y', '\n    double out {};\n    const bool mask1 { ( ( w > 1 ) || ( w < 0 ) ) };\n    if ( mask1 ) {\n        out = nan("0xfff8000000000000ULL");\n    }\n\n    const T tmod { fmod( t, 2.0 * M_PI ) };\n    const bool mask2 { ( ( 1 - mask1 ) && ( tmod < ( w * 2.0 * M_PI ) ) ) };\n\n    if ( mask2 ) {\n        out = tmod / ( M_PI * w ) - 1;\n    }\n\n    const bool mask3 { ( ( 1 - mask1 ) && ( 1 - mask2 ) ) };\n    if ( mask3 ) {\n        out = ( M_PI * ( w + 1 ) - tmod ) / ( M_PI * ( 1 - w ) );\n    }\n    y = out;\n    ', '_sawtooth_kernel', options=('-std=c++11',))

def sawtooth(t, width=1.0):
    if False:
        i = 10
        return i + 15
    '\n    Return a periodic sawtooth or triangle waveform.\n\n    The sawtooth waveform has a period ``2*pi``, rises from -1 to 1 on the\n    interval 0 to ``width*2*pi``, then drops from 1 to -1 on the interval\n    ``width*2*pi`` to ``2*pi``. `width` must be in the interval [0, 1].\n\n    Note that this is not band-limited.  It produces an infinite number\n    of harmonics, which are aliased back and forth across the frequency\n    spectrum.\n\n    Parameters\n    ----------\n    t : array_like\n        Time.\n    width : array_like, optional\n        Width of the rising ramp as a proportion of the total cycle.\n        Default is 1, producing a rising ramp, while 0 produces a falling\n        ramp.  `width` = 0.5 produces a triangle wave.\n        If an array, causes wave shape to change over time, and must be the\n        same length as t.\n\n    Returns\n    -------\n    y : ndarray\n        Output array containing the sawtooth waveform.\n\n    Examples\n    --------\n    A 5 Hz waveform sampled at 500 Hz for 1 second:\n\n    >>> from cupyx.scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> t = np.linspace(0, 1, 500)\n    >>> plt.plot(t, signal.sawtooth(2 * np.pi * 5 * t))\n    '
    (t, w) = (cupy.asarray(t), cupy.asarray(width))
    y = _sawtooth_kernel(t, w)
    return y
_square_kernel = cupy.ElementwiseKernel('T t, T w', 'float64 y', '\n    const bool mask1 { ( ( w > 1 ) || ( w < 0 ) ) };\n    if ( mask1 ) {\n        y = nan("0xfff8000000000000ULL");\n    }\n\n    const T tmod { fmod( t, 2.0 * M_PI ) };\n    const bool mask2 { ( ( 1 - mask1 ) && ( tmod < ( w * 2.0 * M_PI ) ) ) };\n\n    if ( mask2 ) {\n        y = 1;\n    }\n\n    const bool mask3 { ( ( 1 - mask1 ) && ( 1 - mask2 ) ) };\n    if ( mask3 ) {\n        y = -1;\n    }\n\n    ', '_square_kernel', options=('-std=c++11',))

def square(t, duty=0.5):
    if False:
        while True:
            i = 10
    '\n    Return a periodic square-wave waveform.\n\n    The square wave has a period ``2*pi``, has value +1 from 0 to\n    ``2*pi*duty`` and -1 from ``2*pi*duty`` to ``2*pi``. `duty` must be in\n    the interval [0,1].\n\n    Note that this is not band-limited.  It produces an infinite number\n    of harmonics, which are aliased back and forth across the frequency\n    spectrum.\n\n    Parameters\n    ----------\n    t : array_like\n        The input time array.\n    duty : array_like, optional\n        Duty cycle.  Default is 0.5 (50% duty cycle).\n        If an array, causes wave shape to change over time, and must be the\n        same length as t.\n\n    Returns\n    -------\n    y : ndarray\n        Output array containing the square waveform.\n\n    Examples\n    --------\n    A 5 Hz waveform sampled at 500 Hz for 1 second:\n\n    >>> import cupyx.scipy.signal\n    >>> import cupy as cp\n    >>> import matplotlib.pyplot as plt\n    >>> t = cupy.linspace(0, 1, 500, endpoint=False)\n    >>> plt.plot(cupy.asnumpy(t), cupy.asnumpy(cupyx.scipy.signal.square(2 * cupy.pi * 5 * t)))\n    >>> plt.ylim(-2, 2)\n\n    A pulse-width modulated sine wave:\n\n    >>> plt.figure()\n    >>> sig = cupy.sin(2 * cupy.pi * t)\n    >>> pwm = cupyx.scipy.signal.square(2 * cupy.pi * 30 * t, duty=(sig + 1)/2)\n    >>> plt.subplot(2, 1, 1)\n    >>> plt.plot(cupy.asnumpy(t), cupy.asnumpy(sig))\n    >>> plt.subplot(2, 1, 2)\n    >>> plt.plot(cupy.asnumpy(t), cupy.asnumpy(pwm))\n    >>> plt.ylim(-1.5, 1.5)\n\n    '
    (t, w) = (cupy.asarray(t), cupy.asarray(duty))
    y = _square_kernel(t, w)
    return y
_gausspulse_kernel_F_F = cupy.ElementwiseKernel('T t, T a, T fc', 'T yI', '\n    T yenv = exp(-a * t * t);\n    yI = yenv * cos( 2 * M_PI * fc * t);\n    ', '_gausspulse_kernel', options=('-std=c++11',))
_gausspulse_kernel_F_T = cupy.ElementwiseKernel('T t, T a, T fc', 'T yI, T yenv', '\n    yenv = exp(-a * t * t);\n    yI = yenv * cos( 2 * M_PI * fc * t);\n    ', '_gausspulse_kernel', options=('-std=c++11',))
_gausspulse_kernel_T_F = cupy.ElementwiseKernel('T t, T a, T fc', 'T yI, T yQ', '\n    T yenv { exp(-a * t * t) };\n\n    T l_yI {};\n    T l_yQ {};\n    sincos(2 * M_PI * fc * t, &l_yQ, &l_yI);\n    yI = yenv * l_yI;\n    yQ = yenv * l_yQ;\n    ', '_gausspulse_kernel', options=('-std=c++11',))
_gausspulse_kernel_T_T = cupy.ElementwiseKernel('T t, T a, T fc', 'T yI, T yQ, T yenv', '\n    yenv = exp(-a * t * t);\n\n    T l_yI {};\n    T l_yQ {};\n    sincos(2 * M_PI * fc * t, &l_yQ, &l_yI);\n    yI = yenv * l_yI;\n    yQ = yenv * l_yQ;\n    ', '_gausspulse_kernel', options=('-std=c++11',))

def gausspulse(t, fc=1000, bw=0.5, bwr=-6, tpr=-60, retquad=False, retenv=False):
    if False:
        return 10
    "\n    Return a Gaussian modulated sinusoid:\n\n        ``exp(-a t^2) exp(1j*2*pi*fc*t).``\n\n    If `retquad` is True, then return the real and imaginary parts\n    (in-phase and quadrature).\n    If `retenv` is True, then return the envelope (unmodulated signal).\n    Otherwise, return the real part of the modulated sinusoid.\n\n    Parameters\n    ----------\n    t : ndarray or the string 'cutoff'\n        Input array.\n    fc : int, optional\n        Center frequency (e.g. Hz).  Default is 1000.\n    bw : float, optional\n        Fractional bandwidth in frequency domain of pulse (e.g. Hz).\n        Default is 0.5.\n    bwr : float, optional\n        Reference level at which fractional bandwidth is calculated (dB).\n        Default is -6.\n    tpr : float, optional\n        If `t` is 'cutoff', then the function returns the cutoff\n        time for when the pulse amplitude falls below `tpr` (in dB).\n        Default is -60.\n    retquad : bool, optional\n        If True, return the quadrature (imaginary) as well as the real part\n        of the signal.  Default is False.\n    retenv : bool, optional\n        If True, return the envelope of the signal.  Default is False.\n\n    Returns\n    -------\n    yI : ndarray\n        Real part of signal.  Always returned.\n    yQ : ndarray\n        Imaginary part of signal.  Only returned if `retquad` is True.\n    yenv : ndarray\n        Envelope of signal.  Only returned if `retenv` is True.\n\n    See Also\n    --------\n    cupyx.scipy.signal.morlet\n\n    Examples\n    --------\n    Plot real component, imaginary component, and envelope for a 5 Hz pulse,\n    sampled at 100 Hz for 2 seconds:\n\n    >>> import cupyx.scipy.signal\n    >>> import cupy as cp\n    >>> import matplotlib.pyplot as plt\n    >>> t = cupy.linspace(-1, 1, 2 * 100, endpoint=False)\n    >>> i, q, e = cupyx.scipy.signal.gausspulse(t, fc=5, retquad=True, retenv=True)\n    >>> plt.plot(cupy.asnumpy(t), cupy.asnumpy(i), cupy.asnumpy(t), cupy.asnumpy(q),\n                 cupy.asnumpy(t), cupy.asnumpy(e), '--')\n\n    "
    if fc < 0:
        raise ValueError('Center frequency (fc=%.2f) must be >=0.' % fc)
    if bw <= 0:
        raise ValueError('Fractional bandwidth (bw=%.2f) must be > 0.' % bw)
    if bwr >= 0:
        raise ValueError('Reference level for bandwidth (bwr=%.2f) must be < 0 dB' % bwr)
    ref = pow(10.0, bwr / 20.0)
    a = -(np.pi * fc * bw) ** 2 / (4.0 * np.log(ref))
    if isinstance(t, str):
        if t == 'cutoff':
            if tpr >= 0:
                raise ValueError('Reference level for time cutoff must be < 0 dB')
            tref = pow(10.0, tpr / 20.0)
            return np.sqrt(-np.log(tref) / a)
        else:
            raise ValueError("If `t` is a string, it must be 'cutoff'")
    t = cupy.asarray(t)
    if not retquad and (not retenv):
        return _gausspulse_kernel_F_F(t, a, fc)
    if not retquad and retenv:
        return _gausspulse_kernel_F_T(t, a, fc)
    if retquad and (not retenv):
        return _gausspulse_kernel_T_F(t, a, fc)
    if retquad and retenv:
        return _gausspulse_kernel_T_T(t, a, fc)
_chirp_phase_lin_kernel_real = cupy.ElementwiseKernel('T t, T f0, T t1, T f1, T phi', 'T phase', '\n    const T beta { (f1 - f0) / t1 };\n    const T temp { 2 * M_PI * (f0 * t + 0.5 * beta * t * t) };\n    // Convert  phi to radians.\n    phase = cos(temp + phi);\n    ', '_chirp_phase_lin_kernel', options=('-std=c++11',))
_chirp_phase_lin_kernel_cplx = cupy.ElementwiseKernel('T t, T f0, T t1, T f1, T phi', 'Y phase', '\n    const T beta { (f1 - f0) / t1 };\n    const T temp { 2 * M_PI * (f0 * t + 0.5 * beta * t * t) };\n    // Convert  phi to radians.\n    phase = Y(cos(temp + phi), cos(temp + phi + M_PI/2) * -1);\n    ', '_chirp_phase_lin_kernel', options=('-std=c++11',))
_chirp_phase_quad_kernel = cupy.ElementwiseKernel('T t, T f0, T t1, T f1, T phi, bool vertex_zero', 'T phase', '\n    T temp {};\n    const T beta { (f1 - f0) / (t1 * t1) };\n    if ( vertex_zero ) {\n        temp = 2 * M_PI * (f0 * t + beta * (t * t * t) / 3);\n    } else {\n        temp = 2 * M_PI *\n            ( f1 * t + beta *\n            ( ( (t1 - t) * (t1 - t) * (t1 - t) ) - (t1 * t1 * t1)) / 3);\n    }\n    // Convert  phi to radians.\n    phase = cos(temp + phi);\n    ', '_chirp_phase_quad_kernel', options=('-std=c++11',))
_chirp_phase_log_kernel = cupy.ElementwiseKernel('T t, T f0, T t1, T f1, T phi', 'T phase', '\n    T temp {};\n    if ( f0 == f1 ) {\n        temp = 2 * M_PI * f0 * t;\n    } else {\n        T beta { t1 / log(f1 / f0) };\n        temp = 2 * M_PI * beta * f0 * ( pow(f1 / f0, t / t1) - 1.0 );\n    }\n    // Convert  phi to radians.\n    phase = cos(temp + phi);\n    ', '_chirp_phase_log_kernel', options=('-std=c++11',))
_chirp_phase_hyp_kernel = cupy.ElementwiseKernel('T t, T f0, T t1, T f1, T phi', 'T phase', '\n    T temp {};\n    if ( f0 == f1 ) {\n        temp = 2 * M_PI * f0 * t;\n    } else {\n        T sing { -f1 * t1 / (f0 - f1) };\n        temp = 2 * M_PI * ( -sing * f0 ) * log( abs( 1 - t / sing ) );\n    }\n    // Convert  phi to radians.\n    phase = cos(temp + phi);\n    ', '_chirp_phase_hyp_kernel', options=('-std=c++11',))

def chirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True):
    if False:
        return 10
    'Frequency-swept cosine generator.\n\n    In the following, \'Hz\' should be interpreted as \'cycles per unit\';\n    there is no requirement here that the unit is one second.  The\n    important distinction is that the units of rotation are cycles, not\n    radians. Likewise, `t` could be a measurement of space instead of time.\n\n    Parameters\n    ----------\n    t : array_like\n        Times at which to evaluate the waveform.\n    f0 : float\n        Frequency (e.g. Hz) at time t=0.\n    t1 : float\n        Time at which `f1` is specified.\n    f1 : float\n        Frequency (e.g. Hz) of the waveform at time `t1`.\n    method : {\'linear\', \'quadratic\', \'logarithmic\', \'hyperbolic\'}, optional\n        Kind of frequency sweep.  If not given, `linear` is assumed.  See\n        Notes below for more details.\n    phi : float, optional\n        Phase offset, in degrees. Default is 0.\n    vertex_zero : bool, optional\n        This parameter is only used when `method` is \'quadratic\'.\n        It determines whether the vertex of the parabola that is the graph\n        of the frequency is at t=0 or t=t1.\n\n    Returns\n    -------\n    y : ndarray\n        A numpy array containing the signal evaluated at `t` with the\n        requested time-varying frequency.  More precisely, the function\n        returns ``cos(phase + (pi/180)*phi)`` where `phase` is the integral\n        (from 0 to `t`) of ``2*pi*f(t)``. ``f(t)`` is defined below.\n\n    Examples\n    --------\n    The following will be used in the examples:\n\n    >>> from cupyx.scipy.signal import chirp, spectrogram\n    >>> import matplotlib.pyplot as plt\n    >>> import cupy as cp\n\n    For the first example, we\'ll plot the waveform for a linear chirp\n    from 6 Hz to 1 Hz over 10 seconds:\n\n    >>> t = cupy.linspace(0, 10, 5001)\n    >>> w = chirp(t, f0=6, f1=1, t1=10, method=\'linear\')\n    >>> plt.plot(cupy.asnumpy(t), cupy.asnumpy(w))\n    >>> plt.title("Linear Chirp, f(0)=6, f(10)=1")\n    >>> plt.xlabel(\'t (sec)\')\n    >>> plt.show()\n\n    For the remaining examples, we\'ll use higher frequency ranges,\n    and demonstrate the result using `cupyx.scipy.signal.spectrogram`.\n    We\'ll use a 10 second interval sampled at 8000 Hz.\n\n    >>> fs = 8000\n    >>> T = 10\n    >>> t = cupy.linspace(0, T, T*fs, endpoint=False)\n\n    Quadratic chirp from 1500 Hz to 250 Hz over 10 seconds\n    (vertex of the parabolic curve of the frequency is at t=0):\n\n    >>> w = chirp(t, f0=1500, f1=250, t1=10, method=\'quadratic\')\n    >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,\n    ...                           nfft=2048)\n    >>> plt.pcolormesh(cupy.asnumpy(tt), cupy.asnumpy(ff[:513]),\n                       cupy.asnumpy(Sxx[:513]), cmap=\'gray_r\')\n    >>> plt.title(\'Quadratic Chirp, f(0)=1500, f(10)=250\')\n    >>> plt.xlabel(\'t (sec)\')\n    >>> plt.ylabel(\'Frequency (Hz)\')\n    >>> plt.grid()\n    >>> plt.show()\n    '
    t = cupy.asarray(t)
    if cupy.issubdtype(t.dtype, cupy.int_):
        t = t.astype(cupy.float64)
    phi *= np.pi / 180
    type = 'real'
    if method in ['linear', 'lin', 'li']:
        if type == 'real':
            return _chirp_phase_lin_kernel_real(t, f0, t1, f1, phi)
        elif type == 'complex':
            phase = cupy.empty(t.shape, dtype=cupy.complex64)
            if np.issubclass_(t.dtype, np.float64):
                phase = cupy.empty(t.shape, dtype=cupy.complex128)
            _chirp_phase_lin_kernel_cplx(t, f0, t1, f1, phi, phase)
            return phase
        else:
            raise NotImplementedError('No kernel for type {}'.format(type))
    elif method in ['quadratic', 'quad', 'q']:
        return _chirp_phase_quad_kernel(t, f0, t1, f1, phi, vertex_zero)
    elif method in ['logarithmic', 'log', 'lo']:
        if f0 * f1 <= 0.0:
            raise ValueError('For a logarithmic chirp, f0 and f1 must be nonzero and have the same sign.')
        return _chirp_phase_log_kernel(t, f0, t1, f1, phi)
    elif method in ['hyperbolic', 'hyp']:
        if f0 == 0 or f1 == 0:
            raise ValueError('For a hyperbolic chirp, f0 and f1 must be nonzero.')
        return _chirp_phase_hyp_kernel(t, f0, t1, f1, phi)
    else:
        raise ValueError("method must be 'linear', 'quadratic', 'logarithmic', or 'hyperbolic', but a value of %r was given." % method)
if runtime.is_hip:
    KERNEL_BASE = '\n    #include <hip/hip_runtime.h>\n'
else:
    KERNEL_BASE = '\n#include <cuda_runtime.h>\n#include <device_launch_parameters.h>\n'
UNIT_KERNEL = KERNEL_BASE + '\n#include <cupy/math_constants.h>\n#include <cupy/carray.cuh>\n#include <cupy/complex.cuh>\n\n\ntemplate<typename T>\n__global__ void unit_impulse(const int n, const int iidx, T* out) {\n    const int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\n    if(idx >= n) {\n        return;\n    }\n\n    if(idx == iidx) {\n        out[idx] = 1;\n    } else {\n        out[idx] = 0;\n    }\n}\n'
UNIT_MODULE = cupy.RawModule(code=UNIT_KERNEL, options=('-std=c++11',), name_expressions=[f'unit_impulse<{x}>' for x in TYPE_NAMES])

def unit_impulse(shape, idx=None, dtype=float):
    if False:
        i = 10
        return i + 15
    "\n    Unit impulse signal (discrete delta function) or unit basis vector.\n\n    Parameters\n    ----------\n    shape : int or tuple of int\n        Number of samples in the output (1-D), or a tuple that represents the\n        shape of the output (N-D).\n    idx : None or int or tuple of int or 'mid', optional\n        Index at which the value is 1.  If None, defaults to the 0th element.\n        If ``idx='mid'``, the impulse will be centered at ``shape // 2`` in\n        all dimensions.  If an int, the impulse will be at `idx` in all\n        dimensions.\n    dtype : data-type, optional\n        The desired data-type for the array, e.g., ``numpy.int8``.  Default is\n        ``numpy.float64``.\n\n    Returns\n    -------\n    y : ndarray\n        Output array containing an impulse signal.\n\n    Notes\n    -----\n    The 1D case is also known as the Kronecker delta.\n\n    Examples\n    --------\n    An impulse at the 0th element (:math:`\\delta[n]`):\n\n    >>> import cupyx.scipy.signal\n    >>> import cupy as cp\n    >>> cupyx.scipy.signal.unit_impulse(8)\n    array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])\n\n    Impulse offset by 2 samples (:math:`\\delta[n-2]`):\n\n    >>> cupyx.scipy.signal.unit_impulse(7, 2)\n    array([ 0.,  0.,  1.,  0.,  0.,  0.,  0.])\n\n    2-dimensional impulse, centered:\n\n    >>> cupyx.scipy.signal.unit_impulse((3, 3), 'mid')\n    array([[ 0.,  0.,  0.],\n           [ 0.,  1.,  0.],\n           [ 0.,  0.,  0.]])\n\n    Impulse at (2, 2), using broadcasting:\n\n    >>> cupyx.scipy.signal.unit_impulse((4, 4), 2)\n    array([[ 0.,  0.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.],\n           [ 0.,  0.,  1.,  0.],\n           [ 0.,  0.,  0.,  0.]])\n    "
    out = cupy.empty(shape, dtype)
    shape = np.atleast_1d(shape)
    if idx is None:
        idx = (0,) * len(shape)
    elif idx == 'mid':
        idx = tuple(shape // 2)
    elif not hasattr(idx, '__iter__'):
        idx = (idx,) * len(shape)
    pos = np.ravel_multi_index(idx, out.shape)
    n = out.size
    block_sz = 128
    n_blocks = (n + block_sz - 1) // block_sz
    unit_impulse_kernel = _get_module_func(UNIT_MODULE, 'unit_impulse', out)
    unit_impulse_kernel((n_blocks,), (block_sz,), (n, pos, out))
    return out