from numpy import zeros, asarray, eye, poly1d, hstack, r_
from scipy import linalg
__all__ = ['pade']

def pade(an, m, n=None):
    if False:
        i = 10
        return i + 15
    '\n    Return Pade approximation to a polynomial as the ratio of two polynomials.\n\n    Parameters\n    ----------\n    an : (N,) array_like\n        Taylor series coefficients.\n    m : int\n        The order of the returned approximating polynomial `q`.\n    n : int, optional\n        The order of the returned approximating polynomial `p`. By default,\n        the order is ``len(an)-1-m``.\n\n    Returns\n    -------\n    p, q : Polynomial class\n        The Pade approximation of the polynomial defined by `an` is\n        ``p(x)/q(x)``.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.interpolate import pade\n    >>> e_exp = [1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0]\n    >>> p, q = pade(e_exp, 2)\n\n    >>> e_exp.reverse()\n    >>> e_poly = np.poly1d(e_exp)\n\n    Compare ``e_poly(x)`` and the Pade approximation ``p(x)/q(x)``\n\n    >>> e_poly(1)\n    2.7166666666666668\n\n    >>> p(1)/q(1)\n    2.7179487179487181\n\n    '
    an = asarray(an)
    if n is None:
        n = len(an) - 1 - m
        if n < 0:
            raise ValueError('Order of q <m> must be smaller than len(an)-1.')
    if n < 0:
        raise ValueError('Order of p <n> must be greater than 0.')
    N = m + n
    if N > len(an) - 1:
        raise ValueError('Order of q+p <m+n> must be smaller than len(an).')
    an = an[:N + 1]
    Akj = eye(N + 1, n + 1, dtype=an.dtype)
    Bkj = zeros((N + 1, m), dtype=an.dtype)
    for row in range(1, m + 1):
        Bkj[row, :row] = -an[:row][::-1]
    for row in range(m + 1, N + 1):
        Bkj[row, :] = -an[row - m:row][::-1]
    C = hstack((Akj, Bkj))
    pq = linalg.solve(C, an)
    p = pq[:n + 1]
    q = r_[1.0, pq[n + 1:]]
    return (poly1d(p[::-1]), poly1d(q[::-1]))