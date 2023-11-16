"""
Wavelet-generating functions.

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
import numpy as np
from cupyx.scipy.signal._signaltools import convolve
_qmf_kernel = cupy.ElementwiseKernel('raw T coef', 'T output', '\n    const int sign { ( i & 1 ) ? -1 : 1 };\n    output = ( coef[_ind.size() - ( i + 1 )] ) * sign;\n    ', '_qmf_kernel', options=('-std=c++11',))

def qmf(hk):
    if False:
        return 10
    '\n    Return high-pass qmf filter from low-pass\n\n    Parameters\n    ----------\n    hk : array_like\n        Coefficients of high-pass filter.\n\n    '
    hk = cupy.asarray(hk)
    return _qmf_kernel(hk, size=len(hk))
_morlet_kernel = cupy.ElementwiseKernel('float64 w, float64 s, bool complete', 'complex128 output', '\n    const double x { start + delta * i };\n\n    thrust::complex<double> temp { exp(\n        thrust::complex<double>( 0, w * x ) ) };\n\n    if ( complete ) {\n        temp -= exp( -0.5 * ( w * w ) );\n    }\n\n    output = temp * exp( -0.5 * ( x * x ) ) * pow( M_PI, -0.25 )\n    ', '_morlet_kernel', options=('-std=c++11',), loop_prep='const double end { s * 2.0 * M_PI };                const double start { -s * 2.0 * M_PI };                const double delta { ( end - start ) / ( _ind.size() - 1 ) };')

def morlet(M, w=5.0, s=1.0, complete=True):
    if False:
        while True:
            i = 10
    '\n    Complex Morlet wavelet.\n\n    Parameters\n    ----------\n    M : int\n        Length of the wavelet.\n    w : float, optional\n        Omega0. Default is 5\n    s : float, optional\n        Scaling factor, windowed from ``-s*2*pi`` to ``+s*2*pi``. Default is 1.\n    complete : bool, optional\n        Whether to use the complete or the standard version.\n\n    Returns\n    -------\n    morlet : (M,) ndarray\n\n    See Also\n    --------\n    cupyx.scipy.signal.gausspulse\n\n    Notes\n    -----\n    The standard version::\n\n        pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))\n\n    This commonly used wavelet is often referred to simply as the\n    Morlet wavelet.  Note that this simplified version can cause\n    admissibility problems at low values of `w`.\n\n    The complete version::\n\n        pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))\n\n    This version has a correction\n    term to improve admissibility. For `w` greater than 5, the\n    correction term is negligible.\n\n    Note that the energy of the return wavelet is not normalised\n    according to `s`.\n\n    The fundamental frequency of this wavelet in Hz is given\n    by ``f = 2*s*w*r / M`` where `r` is the sampling rate.\n\n    Note: This function was created before `cwt` and is not compatible\n    with it.\n\n    '
    return _morlet_kernel(w, s, complete, size=M)
_ricker_kernel = cupy.ElementwiseKernel('float64 a', 'float64 total', '\n    const double vec { i - ( _ind.size() - 1.0 ) * 0.5 };\n    const double xsq { vec * vec };\n    const double mod { 1 - xsq / wsq };\n    const double gauss { exp( -xsq / ( 2.0 * wsq ) ) };\n\n    total = A * mod * gauss;\n    ', '_ricker_kernel', options=('-std=c++11',), loop_prep='const double A { 2.0 / ( sqrt( 3 * a ) * pow( M_PI, 0.25 ) ) }; const double wsq { a * a };')

def ricker(points, a):
    if False:
        i = 10
        return i + 15
    '\n    Return a Ricker wavelet, also known as the "Mexican hat wavelet".\n\n    It models the function:\n\n        ``A (1 - x^2/a^2) exp(-x^2/2 a^2)``,\n\n    where ``A = 2/sqrt(3a)pi^1/4``.\n\n    Parameters\n    ----------\n    points : int\n        Number of points in `vector`.\n        Will be centered around 0.\n    a : scalar\n        Width parameter of the wavelet.\n\n    Returns\n    -------\n    vector : (N,) ndarray\n        Array of length `points` in shape of ricker curve.\n\n    Examples\n    --------\n    >>> import cupyx.scipy.signal\n    >>> import cupy as cp\n    >>> import matplotlib.pyplot as plt\n\n    >>> points = 100\n    >>> a = 4.0\n    >>> vec2 = cupyx.scipy.signal.ricker(points, a)\n    >>> print(len(vec2))\n    100\n    >>> plt.plot(cupy.asnumpy(vec2))\n    >>> plt.show()\n\n    '
    return _ricker_kernel(a, size=int(points))
_morlet2_kernel = cupy.ElementwiseKernel('float64 w, float64 s', 'complex128 output', '\n    const double x { ( i - ( _ind.size() - 1.0 ) * 0.5 ) / s };\n\n    thrust::complex<double> temp { exp(\n        thrust::complex<double>( 0, w * x ) ) };\n\n    output = sqrt( 1 / s ) * temp * exp( -0.5 * ( x * x ) ) *\n        pow( M_PI, -0.25 )\n    ', '_morlet_kernel', options=('-std=c++11',), loop_prep='')

def morlet2(M, s, w=5):
    if False:
        print('Hello World!')
    "\n    Complex Morlet wavelet, designed to work with `cwt`.\n    Returns the complete version of morlet wavelet, normalised\n    according to `s`::\n\n        exp(1j*w*x/s) * exp(-0.5*(x/s)**2) * pi**(-0.25) * sqrt(1/s)\n\n    Parameters\n    ----------\n    M : int\n        Length of the wavelet.\n    s : float\n        Width parameter of the wavelet.\n    w : float, optional\n        Omega0. Default is 5\n\n    Returns\n    -------\n    morlet : (M,) ndarray\n\n    See Also\n    --------\n    morlet : Implementation of Morlet wavelet, incompatible with `cwt`\n\n    Notes\n    -----\n    This function was designed to work with `cwt`. Because `morlet2`\n    returns an array of complex numbers, the `dtype` argument of `cwt`\n    should be set to `complex128` for best results.\n\n    Note the difference in implementation with `morlet`.\n    The fundamental frequency of this wavelet in Hz is given by::\n\n        f = w*fs / (2*s*np.pi)\n\n    where ``fs`` is the sampling rate and `s` is the wavelet width parameter.\n    Similarly we can get the wavelet width parameter at ``f``::\n\n        s = w*fs / (2*f*np.pi)\n\n    Examples\n    --------\n    >>> from cupyx.scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> M = 100\n    >>> s = 4.0\n    >>> w = 2.0\n    >>> wavelet = signal.morlet2(M, s, w)\n    >>> plt.plot(abs(wavelet))\n    >>> plt.show()\n\n    This example shows basic use of `morlet2` with `cwt` in time-frequency\n    analysis:\n\n    >>> from cupyx.scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> t, dt = np.linspace(0, 1, 200, retstep=True)\n    >>> fs = 1/dt\n    >>> w = 6.\n    >>> sig = np.cos(2*np.pi*(50 + 10*t)*t) + np.sin(40*np.pi*t)\n    >>> freq = np.linspace(1, fs/2, 100)\n    >>> widths = w*fs / (2*freq*np.pi)\n    >>> cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)\n    >>> plt.pcolormesh(t, freq, np.abs(cwtm),\n        cmap='viridis', shading='gouraud')\n    >>> plt.show()\n    "
    return _morlet2_kernel(w, s, size=int(M))

def cwt(data, wavelet, widths):
    if False:
        i = 10
        return i + 15
    "\n    Continuous wavelet transform.\n\n    Performs a continuous wavelet transform on `data`,\n    using the `wavelet` function. A CWT performs a convolution\n    with `data` using the `wavelet` function, which is characterized\n    by a width parameter and length parameter.\n\n    Parameters\n    ----------\n    data : (N,) ndarray\n        data on which to perform the transform.\n    wavelet : function\n        Wavelet function, which should take 2 arguments.\n        The first argument is the number of points that the returned vector\n        will have (len(wavelet(length,width)) == length).\n        The second is a width parameter, defining the size of the wavelet\n        (e.g. standard deviation of a gaussian). See `ricker`, which\n        satisfies these requirements.\n    widths : (M,) sequence\n        Widths to use for transform.\n\n    Returns\n    -------\n    cwt: (M, N) ndarray\n        Will have shape of (len(widths), len(data)).\n\n    Notes\n    -----\n    ::\n\n        length = min(10 * width[ii], len(data))\n        cwt[ii,:] = cupyx.scipy.signal.convolve(data, wavelet(length,\n                                    width[ii]), mode='same')\n\n    Examples\n    --------\n    >>> import cupyx.scipy.signal\n    >>> import cupy as cp\n    >>> import matplotlib.pyplot as plt\n    >>> t = cupy.linspace(-1, 1, 200, endpoint=False)\n    >>> sig  = cupy.cos(2 * cupy.pi * 7 * t) + cupyx.scipy.signal.gausspulse(t - 0.4, fc=2)\n    >>> widths = cupy.arange(1, 31)\n    >>> cwtmatr = cupyx.scipy.signal.cwt(sig, cupyx.scipy.signal.ricker, widths)\n    >>> plt.imshow(abs(cupy.asnumpy(cwtmatr)), extent=[-1, 1, 31, 1],\n                   cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(),\n                   vmin=-abs(cwtmatr).max())\n    >>> plt.show()\n\n    "
    if cupy.asarray(wavelet(1, 1)).dtype.char in 'FDG':
        dtype = cupy.complex128
    else:
        dtype = cupy.float64
    output = cupy.empty([len(widths), len(data)], dtype=dtype)
    for (ind, width) in enumerate(widths):
        N = np.min([10 * int(width), len(data)])
        wavelet_data = cupy.conj(wavelet(N, int(width)))[::-1]
        output[ind, :] = convolve(data, wavelet_data, mode='same')
    return output