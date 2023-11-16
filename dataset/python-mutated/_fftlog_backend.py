import numpy as np
from warnings import warn
from ._basic import rfft, irfft
from ..special import loggamma, poch
from scipy._lib._array_api import array_namespace, copy
__all__ = ['fht', 'ifht', 'fhtoffset']
LN_2 = np.log(2)

def fht(a, dln, mu, offset=0.0, bias=0.0):
    if False:
        print('Hello World!')
    xp = array_namespace(a)
    n = a.shape[-1]
    if bias != 0:
        j_c = (n - 1) / 2
        j = xp.arange(n, dtype=xp.float64)
        a = a * xp.exp(-bias * (j - j_c) * dln)
    u = xp.asarray(fhtcoeff(n, dln, mu, offset=offset, bias=bias))
    A = _fhtq(a, u, xp=xp)
    if bias != 0:
        A *= xp.exp(-bias * ((j - j_c) * dln + offset))
    return A

def ifht(A, dln, mu, offset=0.0, bias=0.0):
    if False:
        for i in range(10):
            print('nop')
    xp = array_namespace(A)
    n = A.shape[-1]
    if bias != 0:
        j_c = (n - 1) / 2
        j = xp.arange(n, dtype=xp.float64)
        A = A * xp.exp(bias * ((j - j_c) * dln + offset))
    u = xp.asarray(fhtcoeff(n, dln, mu, offset=offset, bias=bias, inverse=True))
    a = _fhtq(A, u, inverse=True, xp=xp)
    if bias != 0:
        a /= xp.exp(-bias * (j - j_c) * dln)
    return a

def fhtcoeff(n, dln, mu, offset=0.0, bias=0.0, inverse=False):
    if False:
        i = 10
        return i + 15
    'Compute the coefficient array for a fast Hankel transform.'
    (lnkr, q) = (offset, bias)
    xp = (mu + 1 + q) / 2
    xm = (mu + 1 - q) / 2
    y = np.linspace(0, np.pi * (n // 2) / (n * dln), n // 2 + 1)
    u = np.empty(n // 2 + 1, dtype=complex)
    v = np.empty(n // 2 + 1, dtype=complex)
    u.imag[:] = y
    u.real[:] = xm
    loggamma(u, out=v)
    u.real[:] = xp
    loggamma(u, out=u)
    y *= 2 * (LN_2 - lnkr)
    u.real -= v.real
    u.real += LN_2 * q
    u.imag += v.imag
    u.imag += y
    np.exp(u, out=u)
    u.imag[-1] = 0
    if not np.isfinite(u[0]):
        u[0] = 2 ** q * poch(xm, xp - xm)
    if np.isinf(u[0]) and (not inverse):
        warn('singular transform; consider changing the bias')
        u = copy(u)
        u[0] = 0
    elif u[0] == 0 and inverse:
        warn('singular inverse transform; consider changing the bias')
        u = copy(u)
        u[0] = np.inf
    return u

def fhtoffset(dln, mu, initial=0.0, bias=0.0):
    if False:
        for i in range(10):
            print('nop')
    'Return optimal offset for a fast Hankel transform.\n\n    Returns an offset close to `initial` that fulfils the low-ringing\n    condition of [1]_ for the fast Hankel transform `fht` with logarithmic\n    spacing `dln`, order `mu` and bias `bias`.\n\n    Parameters\n    ----------\n    dln : float\n        Uniform logarithmic spacing of the transform.\n    mu : float\n        Order of the Hankel transform, any positive or negative real number.\n    initial : float, optional\n        Initial value for the offset. Returns the closest value that fulfils\n        the low-ringing condition.\n    bias : float, optional\n        Exponent of power law bias, any positive or negative real number.\n\n    Returns\n    -------\n    offset : float\n        Optimal offset of the uniform logarithmic spacing of the transform that\n        fulfils a low-ringing condition.\n\n    Examples\n    --------\n    >>> from scipy.fft import fhtoffset\n    >>> dln = 0.1\n    >>> mu = 2.0\n    >>> initial = 0.5\n    >>> bias = 0.0\n    >>> offset = fhtoffset(dln, mu, initial, bias)\n    >>> offset\n    0.5454581477676637\n\n    See Also\n    --------\n    fht : Definition of the fast Hankel transform.\n\n    References\n    ----------\n    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)\n\n    '
    (lnkr, q) = (initial, bias)
    xp = (mu + 1 + q) / 2
    xm = (mu + 1 - q) / 2
    y = np.pi / (2 * dln)
    zp = loggamma(xp + 1j * y)
    zm = loggamma(xm + 1j * y)
    arg = (LN_2 - lnkr) / dln + (zp.imag + zm.imag) / np.pi
    return lnkr + (arg - np.round(arg)) * dln

def _fhtq(a, u, inverse=False, *, xp=None):
    if False:
        return 10
    'Compute the biased fast Hankel transform.\n\n    This is the basic FFTLog routine.\n    '
    if xp is None:
        xp = np
    n = a.shape[-1]
    A = rfft(a, axis=-1)
    if not inverse:
        A *= u
    else:
        A /= xp.conj(u)
    A = irfft(A, n, axis=-1)
    A = xp.flip(A, axis=-1)
    return A