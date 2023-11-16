"""
:author: Gary Ruben, 2009
:license: modified BSD
"""
__all__ = ['frt2', 'ifrt2']
import numpy as np
from numpy import roll, newaxis

def frt2(a):
    if False:
        return 10
    'Compute the 2-dimensional finite Radon transform (FRT) for the input array.\n\n    Parameters\n    ----------\n    a : ndarray of int, shape (M, M)\n        Input array.\n\n    Returns\n    -------\n    FRT : ndarray of int, shape (M+1, M)\n        Finite Radon Transform array of coefficients.\n\n    See Also\n    --------\n    ifrt2 : The two-dimensional inverse FRT.\n\n    Notes\n    -----\n    The FRT has a unique inverse if and only if M is prime. [FRT]\n    The idea for this algorithm is due to Vlad Negnevitski.\n\n    Examples\n    --------\n\n    Generate a test image:\n    Use a prime number for the array dimensions\n\n    >>> SIZE = 59\n    >>> img = np.tri(SIZE, dtype=np.int32)\n\n    Apply the Finite Radon Transform:\n\n    >>> f = frt2(img)\n\n    References\n    ----------\n    .. [FRT] A. Kingston and I. Svalbe, "Projective transforms on periodic\n             discrete image arrays," in P. Hawkes (Ed), Advances in Imaging\n             and Electron Physics, 139 (2006)\n\n    '
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('Input must be a square, 2-D array')
    ai = a.copy()
    n = ai.shape[0]
    f = np.empty((n + 1, n), np.uint32)
    f[0] = ai.sum(axis=0)
    for m in range(1, n):
        for row in range(1, n):
            ai[row] = roll(ai[row], -row)
        f[m] = ai.sum(axis=0)
    f[n] = ai.sum(axis=1)
    return f

def ifrt2(a):
    if False:
        print('Hello World!')
    'Compute the 2-dimensional inverse finite Radon transform (iFRT) for the input array.\n\n    Parameters\n    ----------\n    a : ndarray of int, shape (M+1, M)\n        Input array.\n\n    Returns\n    -------\n    iFRT : ndarray of int, shape (M, M)\n        Inverse Finite Radon Transform coefficients.\n\n    See Also\n    --------\n    frt2 : The two-dimensional FRT\n\n    Notes\n    -----\n    The FRT has a unique inverse if and only if M is prime.\n    See [1]_ for an overview.\n    The idea for this algorithm is due to Vlad Negnevitski.\n\n    Examples\n    --------\n\n    >>> SIZE = 59\n    >>> img = np.tri(SIZE, dtype=np.int32)\n\n    Apply the Finite Radon Transform:\n\n    >>> f = frt2(img)\n\n    Apply the Inverse Finite Radon Transform to recover the input\n\n    >>> fi = ifrt2(f)\n\n    Check that it\'s identical to the original\n\n    >>> assert len(np.nonzero(img-fi)[0]) == 0\n\n    References\n    ----------\n    .. [1] A. Kingston and I. Svalbe, "Projective transforms on periodic\n             discrete image arrays," in P. Hawkes (Ed), Advances in Imaging\n             and Electron Physics, 139 (2006)\n\n    '
    if a.ndim != 2 or a.shape[0] != a.shape[1] + 1:
        raise ValueError('Input must be an (n+1) row x n column, 2-D array')
    ai = a.copy()[:-1]
    n = ai.shape[1]
    f = np.empty((n, n), np.uint32)
    f[0] = ai.sum(axis=0)
    for m in range(1, n):
        for row in range(1, ai.shape[0]):
            ai[row] = roll(ai[row], row)
        f[m] = ai.sum(axis=0)
    f += a[-1][newaxis].T
    f = (f - ai[0].sum()) / n
    return f