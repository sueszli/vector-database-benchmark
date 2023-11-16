"""
Differential and pseudo-differential operators.
"""
__all__ = ['diff', 'tilbert', 'itilbert', 'hilbert', 'ihilbert', 'cs_diff', 'cc_diff', 'sc_diff', 'ss_diff', 'shift']
from numpy import pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj
from . import convolve
from scipy.fft._pocketfft.helper import _datacopied
_cache = {}

def diff(x, order=1, period=None, _cache=_cache):
    if False:
        print('Hello World!')
    '\n    Return kth derivative (or integral) of a periodic sequence x.\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = pow(sqrt(-1)*j*2*pi/period, order) * x_j\n      y_0 = 0 if order is not 0.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array.\n    order : int, optional\n        The order of differentiation. Default order is 1. If order is\n        negative, then integration is carried out under the assumption\n        that ``x_0 == 0``.\n    period : float, optional\n        The assumed period of the sequence. Default is ``2*pi``.\n\n    Notes\n    -----\n    If ``sum(x, axis=0) = 0`` then ``diff(diff(x, k), -k) == x`` (within\n    numerical accuracy).\n\n    For odd order and even ``len(x)``, the Nyquist mode is taken zero.\n\n    '
    tmp = asarray(x)
    if order == 0:
        return tmp
    if iscomplexobj(tmp):
        return diff(tmp.real, order, period) + 1j * diff(tmp.imag, order, period)
    if period is not None:
        c = 2 * pi / period
    else:
        c = 1.0
    n = len(x)
    omega = _cache.get((n, order, c))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k, order=order, c=c):
            if False:
                i = 10
                return i + 15
            if k:
                return pow(c * k, order)
            return 0
        omega = convolve.init_convolution_kernel(n, kernel, d=order, zero_nyquist=1)
        _cache[n, order, c] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, swap_real_imag=order % 2, overwrite_x=overwrite_x)
del _cache
_cache = {}

def tilbert(x, h, period=None, _cache=_cache):
    if False:
        return 10
    '\n    Return h-Tilbert transform of a periodic sequence x.\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n        y_j = sqrt(-1)*coth(j*h*2*pi/period) * x_j\n        y_0 = 0\n\n    Parameters\n    ----------\n    x : array_like\n        The input array to transform.\n    h : float\n        Defines the parameter of the Tilbert transform.\n    period : float, optional\n        The assumed period of the sequence. Default period is ``2*pi``.\n\n    Returns\n    -------\n    tilbert : ndarray\n        The result of the transform.\n\n    Notes\n    -----\n    If ``sum(x, axis=0) == 0`` and ``n = len(x)`` is odd, then\n    ``tilbert(itilbert(x)) == x``.\n\n    If ``2 * pi * h / period`` is approximately 10 or larger, then\n    numerically ``tilbert == hilbert``\n    (theoretically oo-Tilbert == Hilbert).\n\n    For even ``len(x)``, the Nyquist mode of ``x`` is taken zero.\n\n    '
    tmp = asarray(x)
    if iscomplexobj(tmp):
        return tilbert(tmp.real, h, period) + 1j * tilbert(tmp.imag, h, period)
    if period is not None:
        h = h * 2 * pi / period
    n = len(x)
    omega = _cache.get((n, h))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k, h=h):
            if False:
                print('Hello World!')
            if k:
                return 1.0 / tanh(h * k)
            return 0
        omega = convolve.init_convolution_kernel(n, kernel, d=1)
        _cache[n, h] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, swap_real_imag=1, overwrite_x=overwrite_x)
del _cache
_cache = {}

def itilbert(x, h, period=None, _cache=_cache):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return inverse h-Tilbert transform of a periodic sequence x.\n\n    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = -sqrt(-1)*tanh(j*h*2*pi/period) * x_j\n      y_0 = 0\n\n    For more details, see `tilbert`.\n\n    '
    tmp = asarray(x)
    if iscomplexobj(tmp):
        return itilbert(tmp.real, h, period) + 1j * itilbert(tmp.imag, h, period)
    if period is not None:
        h = h * 2 * pi / period
    n = len(x)
    omega = _cache.get((n, h))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k, h=h):
            if False:
                while True:
                    i = 10
            if k:
                return -tanh(h * k)
            return 0
        omega = convolve.init_convolution_kernel(n, kernel, d=1)
        _cache[n, h] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, swap_real_imag=1, overwrite_x=overwrite_x)
del _cache
_cache = {}

def hilbert(x, _cache=_cache):
    if False:
        print('Hello World!')
    '\n    Return Hilbert transform of a periodic sequence x.\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = sqrt(-1)*sign(j) * x_j\n      y_0 = 0\n\n    Parameters\n    ----------\n    x : array_like\n        The input array, should be periodic.\n    _cache : dict, optional\n        Dictionary that contains the kernel used to do a convolution with.\n\n    Returns\n    -------\n    y : ndarray\n        The transformed input.\n\n    See Also\n    --------\n    scipy.signal.hilbert : Compute the analytic signal, using the Hilbert\n                           transform.\n\n    Notes\n    -----\n    If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.\n\n    For even len(x), the Nyquist mode of x is taken zero.\n\n    The sign of the returned transform does not have a factor -1 that is more\n    often than not found in the definition of the Hilbert transform. Note also\n    that `scipy.signal.hilbert` does have an extra -1 factor compared to this\n    function.\n\n    '
    tmp = asarray(x)
    if iscomplexobj(tmp):
        return hilbert(tmp.real) + 1j * hilbert(tmp.imag)
    n = len(x)
    omega = _cache.get(n)
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k):
            if False:
                return 10
            if k > 0:
                return 1.0
            elif k < 0:
                return -1.0
            return 0.0
        omega = convolve.init_convolution_kernel(n, kernel, d=1)
        _cache[n] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, swap_real_imag=1, overwrite_x=overwrite_x)
del _cache

def ihilbert(x):
    if False:
        print('Hello World!')
    '\n    Return inverse Hilbert transform of a periodic sequence x.\n\n    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = -sqrt(-1)*sign(j) * x_j\n      y_0 = 0\n\n    '
    return -hilbert(x)
_cache = {}

def cs_diff(x, a, b, period=None, _cache=_cache):
    if False:
        while True:
            i = 10
    '\n    Return (a,b)-cosh/sinh pseudo-derivative of a periodic sequence.\n\n    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = -sqrt(-1)*cosh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j\n      y_0 = 0\n\n    Parameters\n    ----------\n    x : array_like\n        The array to take the pseudo-derivative from.\n    a, b : float\n        Defines the parameters of the cosh/sinh pseudo-differential\n        operator.\n    period : float, optional\n        The period of the sequence. Default period is ``2*pi``.\n\n    Returns\n    -------\n    cs_diff : ndarray\n        Pseudo-derivative of periodic sequence `x`.\n\n    Notes\n    -----\n    For even len(`x`), the Nyquist mode of `x` is taken as zero.\n\n    '
    tmp = asarray(x)
    if iscomplexobj(tmp):
        return cs_diff(tmp.real, a, b, period) + 1j * cs_diff(tmp.imag, a, b, period)
    if period is not None:
        a = a * 2 * pi / period
        b = b * 2 * pi / period
    n = len(x)
    omega = _cache.get((n, a, b))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k, a=a, b=b):
            if False:
                while True:
                    i = 10
            if k:
                return -cosh(a * k) / sinh(b * k)
            return 0
        omega = convolve.init_convolution_kernel(n, kernel, d=1)
        _cache[n, a, b] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, swap_real_imag=1, overwrite_x=overwrite_x)
del _cache
_cache = {}

def sc_diff(x, a, b, period=None, _cache=_cache):
    if False:
        return 10
    '\n    Return (a,b)-sinh/cosh pseudo-derivative of a periodic sequence x.\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = sqrt(-1)*sinh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j\n      y_0 = 0\n\n    Parameters\n    ----------\n    x : array_like\n        Input array.\n    a,b : float\n        Defines the parameters of the sinh/cosh pseudo-differential\n        operator.\n    period : float, optional\n        The period of the sequence x. Default is 2*pi.\n\n    Notes\n    -----\n    ``sc_diff(cs_diff(x,a,b),b,a) == x``\n    For even ``len(x)``, the Nyquist mode of x is taken as zero.\n\n    '
    tmp = asarray(x)
    if iscomplexobj(tmp):
        return sc_diff(tmp.real, a, b, period) + 1j * sc_diff(tmp.imag, a, b, period)
    if period is not None:
        a = a * 2 * pi / period
        b = b * 2 * pi / period
    n = len(x)
    omega = _cache.get((n, a, b))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k, a=a, b=b):
            if False:
                return 10
            if k:
                return sinh(a * k) / cosh(b * k)
            return 0
        omega = convolve.init_convolution_kernel(n, kernel, d=1)
        _cache[n, a, b] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, swap_real_imag=1, overwrite_x=overwrite_x)
del _cache
_cache = {}

def ss_diff(x, a, b, period=None, _cache=_cache):
    if False:
        while True:
            i = 10
    '\n    Return (a,b)-sinh/sinh pseudo-derivative of a periodic sequence x.\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = sinh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j\n      y_0 = a/b * x_0\n\n    Parameters\n    ----------\n    x : array_like\n        The array to take the pseudo-derivative from.\n    a,b\n        Defines the parameters of the sinh/sinh pseudo-differential\n        operator.\n    period : float, optional\n        The period of the sequence x. Default is ``2*pi``.\n\n    Notes\n    -----\n    ``ss_diff(ss_diff(x,a,b),b,a) == x``\n\n    '
    tmp = asarray(x)
    if iscomplexobj(tmp):
        return ss_diff(tmp.real, a, b, period) + 1j * ss_diff(tmp.imag, a, b, period)
    if period is not None:
        a = a * 2 * pi / period
        b = b * 2 * pi / period
    n = len(x)
    omega = _cache.get((n, a, b))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k, a=a, b=b):
            if False:
                i = 10
                return i + 15
            if k:
                return sinh(a * k) / sinh(b * k)
            return float(a) / b
        omega = convolve.init_convolution_kernel(n, kernel)
        _cache[n, a, b] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, overwrite_x=overwrite_x)
del _cache
_cache = {}

def cc_diff(x, a, b, period=None, _cache=_cache):
    if False:
        print('Hello World!')
    '\n    Return (a,b)-cosh/cosh pseudo-derivative of a periodic sequence.\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = cosh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j\n\n    Parameters\n    ----------\n    x : array_like\n        The array to take the pseudo-derivative from.\n    a,b : float\n        Defines the parameters of the sinh/sinh pseudo-differential\n        operator.\n    period : float, optional\n        The period of the sequence x. Default is ``2*pi``.\n\n    Returns\n    -------\n    cc_diff : ndarray\n        Pseudo-derivative of periodic sequence `x`.\n\n    Notes\n    -----\n    ``cc_diff(cc_diff(x,a,b),b,a) == x``\n\n    '
    tmp = asarray(x)
    if iscomplexobj(tmp):
        return cc_diff(tmp.real, a, b, period) + 1j * cc_diff(tmp.imag, a, b, period)
    if period is not None:
        a = a * 2 * pi / period
        b = b * 2 * pi / period
    n = len(x)
    omega = _cache.get((n, a, b))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k, a=a, b=b):
            if False:
                print('Hello World!')
            return cosh(a * k) / cosh(b * k)
        omega = convolve.init_convolution_kernel(n, kernel)
        _cache[n, a, b] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, overwrite_x=overwrite_x)
del _cache
_cache = {}

def shift(x, a, period=None, _cache=_cache):
    if False:
        print('Hello World!')
    '\n    Shift periodic sequence x by a: y(u) = x(u+a).\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n          y_j = exp(j*a*2*pi/period*sqrt(-1)) * x_f\n\n    Parameters\n    ----------\n    x : array_like\n        The array to take the pseudo-derivative from.\n    a : float\n        Defines the parameters of the sinh/sinh pseudo-differential\n    period : float, optional\n        The period of the sequences x and y. Default period is ``2*pi``.\n    '
    tmp = asarray(x)
    if iscomplexobj(tmp):
        return shift(tmp.real, a, period) + 1j * shift(tmp.imag, a, period)
    if period is not None:
        a = a * 2 * pi / period
    n = len(x)
    omega = _cache.get((n, a))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel_real(k, a=a):
            if False:
                i = 10
                return i + 15
            return cos(a * k)

        def kernel_imag(k, a=a):
            if False:
                i = 10
                return i + 15
            return sin(a * k)
        omega_real = convolve.init_convolution_kernel(n, kernel_real, d=0, zero_nyquist=0)
        omega_imag = convolve.init_convolution_kernel(n, kernel_imag, d=1, zero_nyquist=0)
        _cache[n, a] = (omega_real, omega_imag)
    else:
        (omega_real, omega_imag) = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve_z(tmp, omega_real, omega_imag, overwrite_x=overwrite_x)
del _cache