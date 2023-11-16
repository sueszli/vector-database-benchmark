"""
Signal processing B-Splines

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
import cupy
import cupyx.scipy.ndimage
from cupyx.scipy.signal._iir_utils import apply_iir_sos
from cupyx.scipy.signal._splines import _symiirorder1_nd, _symiirorder2_nd
from cupyx.scipy.interpolate._bspline import BSpline
import numpy as np

def sepfir2d(input, hrow, hcol):
    if False:
        print('Hello World!')
    'Convolve with a 2-D separable FIR filter.\n\n    Convolve the rank-2 input array with the separable filter defined by the\n    rank-1 arrays hrow, and hcol. Mirror symmetric boundary conditions are\n    assumed. This function can be used to find an image given its B-spline\n    representation.\n\n    The arguments `hrow` and `hcol` must be 1-dimensional and of off length.\n\n    Args:\n        input (cupy.ndarray): The input signal\n        hrow (cupy.ndarray): Row direction filter\n        hcol (cupy.ndarray): Column direction filter\n\n    Returns:\n        cupy.ndarray: The filtered signal\n\n    .. seealso:: :func:`scipy.signal.sepfir2d`\n    '
    if any((x.ndim != 1 or x.size % 2 == 0 for x in (hrow, hcol))):
        raise ValueError('hrow and hcol must be 1 dimensional and odd length')
    dtype = input.dtype
    if dtype.kind == 'c':
        dtype = cupy.complex64 if dtype == cupy.complex64 else cupy.complex128
    elif dtype == cupy.float32 or dtype.itemsize <= 2:
        dtype = cupy.float32
    else:
        dtype = cupy.float64
    input = input.astype(dtype, copy=False)
    hrow = hrow.astype(dtype, copy=False)
    hcol = hcol.astype(dtype, copy=False)
    filters = (hcol[::-1].conj(), hrow[::-1].conj())
    return cupyx.scipy.ndimage._filters._run_1d_correlates(input, (0, 1), lambda i: filters[i], None, 'reflect', 0)

def _quadratic(x):
    if False:
        print('Hello World!')
    x = abs(cupy.asarray(x, dtype=float))
    b = BSpline.basis_element(cupy.asarray([-1.5, -0.5, 0.5, 1.5]), extrapolate=False)
    out = b(x)
    out[(x < -1.5) | (x > 1.5)] = 0
    return out

def _cubic(x):
    if False:
        for i in range(10):
            print('nop')
    x = cupy.asarray(x, dtype=float)
    b = BSpline.basis_element(cupy.asarray([-2, -1, 0, 1, 2]), extrapolate=False)
    out = b(x)
    out[(x < -2) | (x > 2)] = 0
    return out

@cupy.fuse()
def _coeff_smooth(lam):
    if False:
        while True:
            i = 10
    xi = 1 - 96 * lam + 24 * lam * cupy.sqrt(3 + 144 * lam)
    omeg = cupy.arctan2(cupy.sqrt(144 * lam - 1), cupy.sqrt(xi))
    rho = (24 * lam - 1 - cupy.sqrt(xi)) / (24 * lam)
    rho = rho * cupy.sqrt((48 * lam + 24 * lam * cupy.sqrt(3 + 144 * lam)) / xi)
    return (rho, omeg)

@cupy.fuse()
def _hc(k, cs, rho, omega):
    if False:
        i = 10
        return i + 15
    return cs / cupy.sin(omega) * rho ** k * cupy.sin(omega * (k + 1)) * cupy.greater(k, -1)

@cupy.fuse()
def _hs(k, cs, rho, omega):
    if False:
        i = 10
        return i + 15
    c0 = cs * cs * (1 + rho * rho) / (1 - rho * rho) / (1 - 2 * rho * rho * cupy.cos(2 * omega) + rho ** 4)
    gamma = (1 - rho * rho) / (1 + rho * rho) / cupy.tan(omega)
    ak = cupy.abs(k)
    return c0 * rho ** ak * (cupy.cos(omega * ak) + gamma * cupy.sin(omega * ak))

def _cubic_smooth_coeff(signal, lamb):
    if False:
        return 10
    (rho, omega) = _coeff_smooth(lamb)
    cs = 1 - 2 * rho * cupy.cos(omega) + rho * rho
    K = len(signal)
    yp = cupy.zeros((K,), signal.dtype.char)
    k = cupy.arange(K)
    state_0 = _hc(0, cs, rho, omega) * signal[0] + cupy.sum(_hc(k + 1, cs, rho, omega) * signal)
    state_1 = _hc(0, cs, rho, omega) * signal[0] + _hc(1, cs, rho, omega) * signal[1] + cupy.sum(_hc(k + 2, cs, rho, omega) * signal)
    zi = cupy.r_[0, 0, state_0, state_1]
    zi = cupy.atleast_2d(zi)
    coef = cupy.r_[cs, 0, 0, 1, -2 * rho * cupy.cos(omega), rho * rho]
    coef = cupy.atleast_2d(coef)
    (yp, _) = apply_iir_sos(signal[2:], coef, zi=zi, dtype=signal.dtype)
    yp = cupy.r_[state_0, state_1, yp]
    state_0 = cupy.sum((_hs(k, cs, rho, omega) + _hs(k + 1, cs, rho, omega)) * signal[::-1])
    state_1 = cupy.sum((_hs(k - 1, cs, rho, omega) + _hs(k + 2, cs, rho, omega)) * signal[::-1])
    zi = cupy.r_[0, 0, state_0, state_1]
    zi = cupy.atleast_2d(zi)
    (y, _) = apply_iir_sos(yp[-3::-1], coef, zi=zi, dtype=signal.dtype)
    y = cupy.r_[y[::-1], state_1, state_0]
    return y

def _cubic_coeff(signal):
    if False:
        return 10
    zi = -2 + cupy.sqrt(3)
    K = len(signal)
    powers = zi ** cupy.arange(K)
    if K == 1:
        yplus = signal[0] + zi * cupy.sum(powers * signal)
        output = zi / (zi - 1) * yplus
        return cupy.atleast_1d(output)
    state = cupy.r_[0, 0, 0, cupy.sum(powers * signal)]
    state = cupy.atleast_2d(state)
    coef = cupy.r_[1, 0, 0, 1, -zi, 0]
    coef = cupy.atleast_2d(coef)
    (yplus, _) = apply_iir_sos(signal, coef, zi=state, apply_fir=False, dtype=signal.dtype)
    out_last = zi / (zi - 1) * yplus[K - 1]
    state = cupy.r_[0, 0, 0, out_last]
    state = cupy.atleast_2d(state)
    coef = cupy.r_[-zi, 0, 0, 1, -zi, 0]
    coef = cupy.atleast_2d(coef)
    (output, _) = apply_iir_sos(yplus[-2::-1], coef, zi=state, dtype=signal.dtype)
    output = cupy.r_[output[::-1], out_last]
    return output * 6.0

def _quadratic_coeff(signal):
    if False:
        while True:
            i = 10
    zi = -3 + 2 * cupy.sqrt(2.0)
    K = len(signal)
    powers = zi ** cupy.arange(K)
    if K == 1:
        yplus = signal[0] + zi * cupy.sum(powers * signal)
        output = zi / (zi - 1) * yplus
        return cupy.atleast_1d(output)
    state = cupy.r_[0, 0, 0, cupy.sum(powers * signal)]
    state = cupy.atleast_2d(state)
    coef = cupy.r_[1, 0, 0, 1, -zi, 0]
    coef = cupy.atleast_2d(coef)
    (yplus, _) = apply_iir_sos(signal, coef, zi=state, apply_fir=False, dtype=signal.dtype)
    out_last = zi / (zi - 1) * yplus[K - 1]
    state = cupy.r_[0, 0, 0, out_last]
    state = cupy.atleast_2d(state)
    coef = cupy.r_[-zi, 0, 0, 1, -zi, 0]
    coef = cupy.atleast_2d(coef)
    (output, _) = apply_iir_sos(yplus[-2::-1], coef, zi=state, dtype=signal.dtype)
    output = cupy.r_[output[::-1], out_last]
    return output * 8.0

def compute_root_from_lambda(lamb):
    if False:
        for i in range(10):
            print('nop')
    tmp = np.sqrt(3 + 144 * lamb)
    xi = 1 - 96 * lamb + 24 * lamb * tmp
    omega = np.arctan(np.sqrt((144 * lamb - 1.0) / xi))
    tmp2 = np.sqrt(xi)
    r = (24 * lamb - 1 - tmp2) / (24 * lamb) * np.sqrt(48 * lamb + 24 * lamb * tmp) / tmp2
    return (r, omega)

def cspline1d(signal, lamb=0.0):
    if False:
        return 10
    '\n    Compute cubic spline coefficients for rank-1 array.\n\n    Find the cubic spline coefficients for a 1-D signal assuming\n    mirror-symmetric boundary conditions. To obtain the signal back from the\n    spline representation mirror-symmetric-convolve these coefficients with a\n    length 3 FIR window [1.0, 4.0, 1.0]/ 6.0 .\n\n    Parameters\n    ----------\n    signal : ndarray\n        A rank-1 array representing samples of a signal.\n    lamb : float, optional\n        Smoothing coefficient, default is 0.0.\n\n    Returns\n    -------\n    c : ndarray\n        Cubic spline coefficients.\n\n    See Also\n    --------\n    cspline1d_eval : Evaluate a cubic spline at the new set of points.\n\n    '
    if lamb != 0.0:
        return _cubic_smooth_coeff(signal, lamb)
    else:
        return _cubic_coeff(signal)

def qspline1d(signal, lamb=0.0):
    if False:
        print('Hello World!')
    'Compute quadratic spline coefficients for rank-1 array.\n\n    Parameters\n    ----------\n    signal : ndarray\n        A rank-1 array representing samples of a signal.\n    lamb : float, optional\n        Smoothing coefficient (must be zero for now).\n\n    Returns\n    -------\n    c : ndarray\n        Quadratic spline coefficients.\n\n    See Also\n    --------\n    qspline1d_eval : Evaluate a quadratic spline at the new set of points.\n\n    Notes\n    -----\n    Find the quadratic spline coefficients for a 1-D signal assuming\n    mirror-symmetric boundary conditions. To obtain the signal back from the\n    spline representation mirror-symmetric-convolve these coefficients with a\n    length 3 FIR window [1.0, 6.0, 1.0]/ 8.0 .\n\n    '
    if lamb != 0.0:
        raise ValueError('Smoothing quadratic splines not supported yet.')
    else:
        return _quadratic_coeff(signal)

def cspline1d_eval(cj, newx, dx=1.0, x0=0):
    if False:
        print('Hello World!')
    'Evaluate a cubic spline at the new set of points.\n\n    `dx` is the old sample-spacing while `x0` was the old origin. In\n    other-words the old-sample points (knot-points) for which the `cj`\n    represent spline coefficients were at equally-spaced points of:\n\n      oldx = x0 + j*dx  j=0...N-1, with N=len(cj)\n\n    Edges are handled using mirror-symmetric boundary conditions.\n\n    Parameters\n    ----------\n    cj : ndarray\n        cublic spline coefficients\n    newx : ndarray\n        New set of points.\n    dx : float, optional\n        Old sample-spacing, the default value is 1.0.\n    x0 : int, optional\n        Old origin, the default value is 0.\n\n    Returns\n    -------\n    res : ndarray\n        Evaluated a cubic spline points.\n\n    See Also\n    --------\n    cspline1d : Compute cubic spline coefficients for rank-1 array.\n\n    '
    newx = (cupy.asarray(newx) - x0) / float(dx)
    res = cupy.zeros_like(newx, dtype=cj.dtype)
    if res.size == 0:
        return res
    N = len(cj)
    cond1 = newx < 0
    cond2 = newx > N - 1
    cond3 = ~(cond1 | cond2)
    res[cond1] = cspline1d_eval(cj, -newx[cond1])
    res[cond2] = cspline1d_eval(cj, 2 * (N - 1) - newx[cond2])
    newx = newx[cond3]
    if newx.size == 0:
        return res
    result = cupy.zeros_like(newx, dtype=cj.dtype)
    jlower = cupy.floor(newx - 2).astype(int) + 1
    for i in range(4):
        thisj = jlower + i
        indj = thisj.clip(0, N - 1)
        result += cj[indj] * _cubic(newx - thisj)
    res[cond3] = result
    return res

def qspline1d_eval(cj, newx, dx=1.0, x0=0):
    if False:
        while True:
            i = 10
    'Evaluate a quadratic spline at the new set of points.\n\n    Parameters\n    ----------\n    cj : ndarray\n        Quadratic spline coefficients\n    newx : ndarray\n        New set of points.\n    dx : float, optional\n        Old sample-spacing, the default value is 1.0.\n    x0 : int, optional\n        Old origin, the default value is 0.\n\n    Returns\n    -------\n    res : ndarray\n        Evaluated a quadratic spline points.\n\n    See Also\n    --------\n    qspline1d : Compute quadratic spline coefficients for rank-1 array.\n\n    Notes\n    -----\n    `dx` is the old sample-spacing while `x0` was the old origin. In\n    other-words the old-sample points (knot-points) for which the `cj`\n    represent spline coefficients were at equally-spaced points of::\n\n      oldx = x0 + j*dx  j=0...N-1, with N=len(cj)\n\n    Edges are handled using mirror-symmetric boundary conditions.\n\n    '
    newx = (cupy.asarray(newx) - x0) / dx
    res = cupy.zeros_like(newx)
    if res.size == 0:
        return res
    N = len(cj)
    cond1 = newx < 0
    cond2 = newx > N - 1
    cond3 = ~(cond1 | cond2)
    res[cond1] = qspline1d_eval(cj, -newx[cond1])
    res[cond2] = qspline1d_eval(cj, 2 * (N - 1) - newx[cond2])
    newx = newx[cond3]
    if newx.size == 0:
        return res
    result = cupy.zeros_like(newx)
    jlower = cupy.floor(newx - 1.5).astype(int) + 1
    for i in range(3):
        thisj = jlower + i
        indj = thisj.clip(0, N - 1)
        result += cj[indj] * _quadratic(newx - thisj)
    res[cond3] = result
    return res

def cspline2d(signal, lamb=0.0, precision=-1.0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Coefficients for 2-D cubic (3rd order) B-spline.\n\n    Return the third-order B-spline coefficients over a regularly spaced\n    input grid for the two-dimensional input image.\n\n    Parameters\n    ----------\n    input : ndarray\n        The input signal.\n    lamb : float\n        Specifies the amount of smoothing in the transfer function.\n    precision : float\n        Specifies the precision for computing the infinite sum needed to apply\n        mirror-symmetric boundary conditions.\n\n    Returns\n    -------\n    output : ndarray\n        The filtered signal.\n    '
    if lamb <= 1 / 144.0:
        r = -2 + np.sqrt(3.0)
        out = _symiirorder1_nd(signal, -r * 6.0, r, precision=precision, axis=-1)
        out = _symiirorder1_nd(out, -r * 6.0, r, precision=precision, axis=0)
        return out
    (r, omega) = compute_root_from_lambda(lamb)
    out = _symiirorder2_nd(signal, r, omega, precision=precision, axis=-1)
    out = _symiirorder2_nd(out, r, omega, precision=precision, axis=0)
    return out

def qspline2d(signal, lamb=0.0, precision=-1.0):
    if False:
        return 10
    '\n    Coefficients for 2-D quadratic (2nd order) B-spline.\n\n    Return the second-order B-spline coefficients over a regularly spaced\n    input grid for the two-dimensional input image.\n\n    Parameters\n    ----------\n    input : ndarray\n        The input signal.\n    lamb : float\n        Specifies the amount of smoothing in the transfer function.\n    precision : float\n        Specifies the precision for computing the infinite sum needed to apply\n        mirror-symmetric boundary conditions.\n\n    Returns\n    -------\n    output : ndarray\n        The filtered signal.\n    '
    if lamb > 0:
        raise ValueError('lambda must be negative or zero')
    r = -3 + 2 * np.sqrt(2.0)
    out = _symiirorder1_nd(signal, -r * 8.0, r, precision=precision, axis=-1)
    out = _symiirorder1_nd(out, -r * 8.0, r, precision=precision, axis=0)
    return out

def spline_filter(Iin, lmbda=5.0):
    if False:
        return 10
    'Smoothing spline (cubic) filtering of a rank-2 array.\n\n    Filter an input data set, `Iin`, using a (cubic) smoothing spline of\n    fall-off `lmbda`.\n\n    Parameters\n    ----------\n    Iin : array_like\n        input data set\n    lmbda : float, optional\n        spline smooghing fall-off value, default is `5.0`.\n\n    Returns\n    -------\n    res : ndarray\n        filterd input data\n\n    '
    intype = Iin.dtype.char
    hcol = cupy.asarray([1.0, 4.0, 1.0], 'f') / 6.0
    if intype in ['F', 'D']:
        Iin = Iin.astype('F')
        ckr = cspline2d(Iin.real, lmbda)
        cki = cspline2d(Iin.imag, lmbda)
        outr = sepfir2d(ckr, hcol, hcol)
        outi = sepfir2d(cki, hcol, hcol)
        out = (outr + 1j * outi).astype(intype)
    elif intype in ['f', 'd']:
        ckr = cspline2d(Iin, lmbda)
        out = sepfir2d(ckr, hcol, hcol)
        out = out.astype(intype)
    else:
        raise TypeError('Invalid data type for Iin')
    return out
_gauss_spline_kernel = cupy.ElementwiseKernel('T x, int32 n', 'T output', '\n    output = 1 / sqrt( 2.0 * M_PI * signsq ) * exp( -( x * x ) * r_signsq );\n    ', '_gauss_spline_kernel', options=('-std=c++11',), loop_prep='const double signsq { ( n + 1 ) / 12.0 };                const double r_signsq { 0.5 / signsq };')

def gauss_spline(x, n):
    if False:
        return 10
    'Gaussian approximation to B-spline basis function of order n.\n\n    Parameters\n    ----------\n    x : array_like\n        a knot vector\n    n : int\n        The order of the spline. Must be nonnegative, i.e. n >= 0\n\n    Returns\n    -------\n    res : ndarray\n        B-spline basis function values approximated by a zero-mean Gaussian\n        function.\n\n    Notes\n    -----\n    The B-spline basis function can be approximated well by a zero-mean\n    Gaussian function with standard-deviation equal to :math:`\\sigma=(n+1)/12`\n    for large `n` :\n\n    .. math::  \\frac{1}{\\sqrt {2\\pi\\sigma^2}}exp(-\\frac{x^2}{2\\sigma})\n\n    See [1]_, [2]_ for more information.\n\n    References\n    ----------\n    .. [1] Bouma H., Vilanova A., Bescos J.O., ter Haar Romeny B.M., Gerritsen\n       F.A. (2007) Fast and Accurate Gaussian Derivatives Based on B-Splines.\n       In: Sgallari F., Murli A., Paragios N. (eds) Scale Space and Variational\n       Methods in Computer Vision. SSVM 2007. Lecture Notes in Computer\n       Science, vol 4485. Springer, Berlin, Heidelberg\n    .. [2] http://folk.uio.no/inf3330/scripting/doc/python/SciPy/tutorial/old/node24.html\n    '
    x = cupy.asarray(x)
    return _gauss_spline_kernel(x, n)