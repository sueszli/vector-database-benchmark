import numpy as np
from scipy.linalg import svd
__all__ = ['polar']

def polar(a, side='right'):
    if False:
        while True:
            i = 10
    '\n    Compute the polar decomposition.\n\n    Returns the factors of the polar decomposition [1]_ `u` and `p` such\n    that ``a = up`` (if `side` is "right") or ``a = pu`` (if `side` is\n    "left"), where `p` is positive semidefinite. Depending on the shape\n    of `a`, either the rows or columns of `u` are orthonormal. When `a`\n    is a square array, `u` is a square unitary array. When `a` is not\n    square, the "canonical polar decomposition" [2]_ is computed.\n\n    Parameters\n    ----------\n    a : (m, n) array_like\n        The array to be factored.\n    side : {\'left\', \'right\'}, optional\n        Determines whether a right or left polar decomposition is computed.\n        If `side` is "right", then ``a = up``.  If `side` is "left",  then\n        ``a = pu``.  The default is "right".\n\n    Returns\n    -------\n    u : (m, n) ndarray\n        If `a` is square, then `u` is unitary. If m > n, then the columns\n        of `a` are orthonormal, and if m < n, then the rows of `u` are\n        orthonormal.\n    p : ndarray\n        `p` is Hermitian positive semidefinite. If `a` is nonsingular, `p`\n        is positive definite. The shape of `p` is (n, n) or (m, m), depending\n        on whether `side` is "right" or "left", respectively.\n\n    References\n    ----------\n    .. [1] R. A. Horn and C. R. Johnson, "Matrix Analysis", Cambridge\n           University Press, 1985.\n    .. [2] N. J. Higham, "Functions of Matrices: Theory and Computation",\n           SIAM, 2008.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import polar\n    >>> a = np.array([[1, -1], [2, 4]])\n    >>> u, p = polar(a)\n    >>> u\n    array([[ 0.85749293, -0.51449576],\n           [ 0.51449576,  0.85749293]])\n    >>> p\n    array([[ 1.88648444,  1.2004901 ],\n           [ 1.2004901 ,  3.94446746]])\n\n    A non-square example, with m < n:\n\n    >>> b = np.array([[0.5, 1, 2], [1.5, 3, 4]])\n    >>> u, p = polar(b)\n    >>> u\n    array([[-0.21196618, -0.42393237,  0.88054056],\n           [ 0.39378971,  0.78757942,  0.4739708 ]])\n    >>> p\n    array([[ 0.48470147,  0.96940295,  1.15122648],\n           [ 0.96940295,  1.9388059 ,  2.30245295],\n           [ 1.15122648,  2.30245295,  3.65696431]])\n    >>> u.dot(p)   # Verify the decomposition.\n    array([[ 0.5,  1. ,  2. ],\n           [ 1.5,  3. ,  4. ]])\n    >>> u.dot(u.T)   # The rows of u are orthonormal.\n    array([[  1.00000000e+00,  -2.07353665e-17],\n           [ -2.07353665e-17,   1.00000000e+00]])\n\n    Another non-square example, with m > n:\n\n    >>> c = b.T\n    >>> u, p = polar(c)\n    >>> u\n    array([[-0.21196618,  0.39378971],\n           [-0.42393237,  0.78757942],\n           [ 0.88054056,  0.4739708 ]])\n    >>> p\n    array([[ 1.23116567,  1.93241587],\n           [ 1.93241587,  4.84930602]])\n    >>> u.dot(p)   # Verify the decomposition.\n    array([[ 0.5,  1.5],\n           [ 1. ,  3. ],\n           [ 2. ,  4. ]])\n    >>> u.T.dot(u)  # The columns of u are orthonormal.\n    array([[  1.00000000e+00,  -1.26363763e-16],\n           [ -1.26363763e-16,   1.00000000e+00]])\n\n    '
    if side not in ['right', 'left']:
        raise ValueError("`side` must be either 'right' or 'left'")
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError('`a` must be a 2-D array.')
    (w, s, vh) = svd(a, full_matrices=False)
    u = w.dot(vh)
    if side == 'right':
        p = (vh.T.conj() * s).dot(vh)
    else:
        p = (w * s).dot(w.T.conj())
    return (u, p)