"""Fast Hankel transforms using the FFTLog algorithm.
The implementation closely follows the Fortran code of Hamilton (2000).
"""
import math
from warnings import warn
import cupy
from cupyx.scipy.fft import _fft
from cupyx.scipy.special import loggamma, poch
try:
    from scipy.fft import fht as _fht
    _scipy_fft = _fft._scipy_fft
    del _fht
except ImportError:

    class _DummyModule:

        def __getattr__(self, name):
            if False:
                while True:
                    i = 10
            return None
    _scipy_fft = _DummyModule()
__all__ = ['fht', 'ifht']
LN_2 = math.log(2)

@_fft._implements(_scipy_fft.fht)
def fht(a, dln, mu, offset=0.0, bias=0.0):
    if False:
        while True:
            i = 10
    'Compute the fast Hankel transform.\n\n    Computes the discrete Hankel transform of a logarithmically spaced periodic\n    sequence using the FFTLog algorithm [1]_, [2]_.\n\n    Parameters\n    ----------\n    a : cupy.ndarray (..., n)\n        Real periodic input array, uniformly logarithmically spaced.  For\n        multidimensional input, the transform is performed over the last axis.\n    dln : float\n        Uniform logarithmic spacing of the input array.\n    mu : float\n        Order of the Hankel transform, any positive or negative real number.\n    offset : float, optional\n        Offset of the uniform logarithmic spacing of the output array.\n    bias : float, optional\n        Exponent of power law bias, any positive or negative real number.\n\n    Returns\n    -------\n    A : cupy.ndarray (..., n)\n        The transformed output array, which is real, periodic, uniformly\n        logarithmically spaced, and of the same shape as the input array.\n\n    See Also\n    --------\n    :func:`scipy.special.fht`\n    :func:`scipy.special.fhtoffset` : Return an optimal offset for `fht`.\n\n    References\n    ----------\n    .. [1] Talman J. D., 1978, J. Comp. Phys., 29, 35\n    .. [2] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)\n\n    '
    n = a.shape[-1]
    if bias != 0:
        j_c = (n - 1) / 2
        j = cupy.arange(n)
        a = a * cupy.exp(-bias * (j - j_c) * dln)
    u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)
    A = _fhtq(a, u)
    if bias != 0:
        A *= cupy.exp(-bias * ((j - j_c) * dln + offset))
    return A

@_fft._implements(_scipy_fft.ifht)
def ifht(A, dln, mu, offset=0.0, bias=0.0):
    if False:
        print('Hello World!')
    'Compute the inverse fast Hankel transform.\n\n    Computes the discrete inverse Hankel transform of a logarithmically spaced\n    periodic sequence. This is the inverse operation to `fht`.\n\n    Parameters\n    ----------\n    A : cupy.ndarray (..., n)\n        Real periodic input array, uniformly logarithmically spaced.  For\n        multidimensional input, the transform is performed over the last axis.\n    dln : float\n        Uniform logarithmic spacing of the input array.\n    mu : float\n        Order of the Hankel transform, any positive or negative real number.\n    offset : float, optional\n        Offset of the uniform logarithmic spacing of the output array.\n    bias : float, optional\n        Exponent of power law bias, any positive or negative real number.\n\n    Returns\n    -------\n    a : cupy.ndarray (..., n)\n        The transformed output array, which is real, periodic, uniformly\n        logarithmically spaced, and of the same shape as the input array.\n\n    See Also\n    --------\n    :func:`scipy.special.ifht`\n    :func:`scipy.special.fhtoffset` : Return an optimal offset for `fht`.\n\n    '
    n = A.shape[-1]
    if bias != 0:
        j_c = (n - 1) / 2
        j = cupy.arange(n)
        A = A * cupy.exp(bias * ((j - j_c) * dln + offset))
    u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)
    a = _fhtq(A, u, inverse=True)
    if bias != 0:
        a /= cupy.exp(-bias * (j - j_c) * dln)
    return a

def fhtcoeff(n, dln, mu, offset=0.0, bias=0.0):
    if False:
        for i in range(10):
            print('nop')
    'Compute the coefficient array for a fast Hankel transform.\n    '
    (lnkr, q) = (offset, bias)
    xp = (mu + 1 + q) / 2
    xm = (mu + 1 - q) / 2
    y = cupy.linspace(0, math.pi * (n // 2) / (n * dln), n // 2 + 1)
    u = cupy.empty(n // 2 + 1, dtype=complex)
    v = cupy.empty(n // 2 + 1, dtype=complex)
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
    cupy.exp(u, out=u)
    u.imag[-1] = 0
    if not cupy.isfinite(u[0]):
        u[0] = 2 ** q * poch(xm, xp - xm)
    return u

def _fhtq(a, u, inverse=False):
    if False:
        for i in range(10):
            print('nop')
    'Compute the biased fast Hankel transform.\n\n    This is the basic FFTLog routine.\n    '
    n = a.shape[-1]
    if cupy.isinf(u[0]) and (not inverse):
        warn('singular transform; consider changing the bias')
        u = u.copy()
        u[0] = 0
    elif u[0] == 0 and inverse:
        warn('singular inverse transform; consider changing the bias')
        u = u.copy()
        u[0] = cupy.inf
    A = _fft.rfft(a, axis=-1)
    if not inverse:
        A *= u
    else:
        A /= u.conj()
    A = _fft.irfft(A, n, axis=-1)
    A = A[..., ::-1]
    return A