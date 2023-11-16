"""Cholesky decomposition functions."""
from numpy import asarray_chkfinite, asarray, atleast_2d
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs
__all__ = ['cholesky', 'cho_factor', 'cho_solve', 'cholesky_banded', 'cho_solve_banded']

def _cholesky(a, lower=False, overwrite_a=False, clean=True, check_finite=True):
    if False:
        print('Hello World!')
    'Common code for cholesky() and cho_factor().'
    a1 = asarray_chkfinite(a) if check_finite else asarray(a)
    a1 = atleast_2d(a1)
    if a1.ndim != 2:
        raise ValueError('Input array needs to be 2D but received a {}d-array.'.format(a1.ndim))
    if a1.shape[0] != a1.shape[1]:
        raise ValueError('Input array is expected to be square but has the shape: {}.'.format(a1.shape))
    if a1.size == 0:
        return (a1.copy(), lower)
    overwrite_a = overwrite_a or _datacopied(a1, a)
    (potrf,) = get_lapack_funcs(('potrf',), (a1,))
    (c, info) = potrf(a1, lower=lower, overwrite_a=overwrite_a, clean=clean)
    if info > 0:
        raise LinAlgError('%d-th leading minor of the array is not positive definite' % info)
    if info < 0:
        raise ValueError('LAPACK reported an illegal value in {}-th argumenton entry to "POTRF".'.format(-info))
    return (c, lower)

def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    if False:
        return 10
    '\n    Compute the Cholesky decomposition of a matrix.\n\n    Returns the Cholesky decomposition, :math:`A = L L^*` or\n    :math:`A = U^* U` of a Hermitian positive-definite matrix A.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Matrix to be decomposed\n    lower : bool, optional\n        Whether to compute the upper- or lower-triangular Cholesky\n        factorization.  Default is upper-triangular.\n    overwrite_a : bool, optional\n        Whether to overwrite data in `a` (may improve performance).\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    c : (M, M) ndarray\n        Upper- or lower-triangular Cholesky factor of `a`.\n\n    Raises\n    ------\n    LinAlgError : if decomposition fails.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import cholesky\n    >>> a = np.array([[1,-2j],[2j,5]])\n    >>> L = cholesky(a, lower=True)\n    >>> L\n    array([[ 1.+0.j,  0.+0.j],\n           [ 0.+2.j,  1.+0.j]])\n    >>> L @ L.T.conj()\n    array([[ 1.+0.j,  0.-2.j],\n           [ 0.+2.j,  5.+0.j]])\n\n    '
    (c, lower) = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=True, check_finite=check_finite)
    return c

def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the Cholesky decomposition of a matrix, to use in cho_solve\n\n    Returns a matrix containing the Cholesky decomposition,\n    ``A = L L*`` or ``A = U* U`` of a Hermitian positive-definite matrix `a`.\n    The return value can be directly used as the first parameter to cho_solve.\n\n    .. warning::\n        The returned matrix also contains random data in the entries not\n        used by the Cholesky decomposition. If you need to zero these\n        entries, use the function `cholesky` instead.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Matrix to be decomposed\n    lower : bool, optional\n        Whether to compute the upper or lower triangular Cholesky factorization\n        (Default: upper-triangular)\n    overwrite_a : bool, optional\n        Whether to overwrite data in a (may improve performance)\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    c : (M, M) ndarray\n        Matrix whose upper or lower triangle contains the Cholesky factor\n        of `a`. Other parts of the matrix contain random data.\n    lower : bool\n        Flag indicating whether the factor is in the lower or upper triangle\n\n    Raises\n    ------\n    LinAlgError\n        Raised if decomposition fails.\n\n    See Also\n    --------\n    cho_solve : Solve a linear set equations using the Cholesky factorization\n                of a matrix.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import cho_factor\n    >>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])\n    >>> c, low = cho_factor(A)\n    >>> c\n    array([[3.        , 1.        , 0.33333333, 1.66666667],\n           [3.        , 2.44948974, 1.90515869, -0.27216553],\n           [1.        , 5.        , 2.29330749, 0.8559528 ],\n           [5.        , 1.        , 2.        , 1.55418563]])\n    >>> np.allclose(np.triu(c).T @ np. triu(c) - A, np.zeros((4, 4)))\n    True\n\n    '
    (c, lower) = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=False, check_finite=check_finite)
    return (c, lower)

def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    if False:
        while True:
            i = 10
    'Solve the linear equations A x = b, given the Cholesky factorization of A.\n\n    Parameters\n    ----------\n    (c, lower) : tuple, (array, bool)\n        Cholesky factorization of a, as given by cho_factor\n    b : array\n        Right-hand side\n    overwrite_b : bool, optional\n        Whether to overwrite data in b (may improve performance)\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    x : array\n        The solution to the system A x = b\n\n    See Also\n    --------\n    cho_factor : Cholesky factorization of a matrix\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import cho_factor, cho_solve\n    >>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])\n    >>> c, low = cho_factor(A)\n    >>> x = cho_solve((c, low), [1, 1, 1, 1])\n    >>> np.allclose(A @ x - [1, 1, 1, 1], np.zeros(4))\n    True\n\n    '
    (c, lower) = c_and_lower
    if check_finite:
        b1 = asarray_chkfinite(b)
        c = asarray_chkfinite(c)
    else:
        b1 = asarray(b)
        c = asarray(c)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError('The factored matrix c is not square.')
    if c.shape[1] != b1.shape[0]:
        raise ValueError('incompatible dimensions ({} and {})'.format(c.shape, b1.shape))
    overwrite_b = overwrite_b or _datacopied(b1, b)
    (potrs,) = get_lapack_funcs(('potrs',), (c, b1))
    (x, info) = potrs(c, b1, lower=lower, overwrite_b=overwrite_b)
    if info != 0:
        raise ValueError('illegal value in %dth argument of internal potrs' % -info)
    return x

def cholesky_banded(ab, overwrite_ab=False, lower=False, check_finite=True):
    if False:
        return 10
    '\n    Cholesky decompose a banded Hermitian positive-definite matrix\n\n    The matrix a is stored in ab either in lower-diagonal or upper-\n    diagonal ordered form::\n\n        ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)\n        ab[    i - j, j] == a[i,j]        (if lower form; i >= j)\n\n    Example of ab (shape of a is (6,6), u=2)::\n\n        upper form:\n        *   *   a02 a13 a24 a35\n        *   a01 a12 a23 a34 a45\n        a00 a11 a22 a33 a44 a55\n\n        lower form:\n        a00 a11 a22 a33 a44 a55\n        a10 a21 a32 a43 a54 *\n        a20 a31 a42 a53 *   *\n\n    Parameters\n    ----------\n    ab : (u + 1, M) array_like\n        Banded matrix\n    overwrite_ab : bool, optional\n        Discard data in ab (may enhance performance)\n    lower : bool, optional\n        Is the matrix in the lower form. (Default is upper form)\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    c : (u + 1, M) ndarray\n        Cholesky factorization of a, in the same banded format as ab\n\n    See Also\n    --------\n    cho_solve_banded :\n        Solve a linear set equations, given the Cholesky factorization\n        of a banded Hermitian.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import cholesky_banded\n    >>> from numpy import allclose, zeros, diag\n    >>> Ab = np.array([[0, 0, 1j, 2, 3j], [0, -1, -2, 3, 4], [9, 8, 7, 6, 9]])\n    >>> A = np.diag(Ab[0,2:], k=2) + np.diag(Ab[1,1:], k=1)\n    >>> A = A + A.conj().T + np.diag(Ab[2, :])\n    >>> c = cholesky_banded(Ab)\n    >>> C = np.diag(c[0, 2:], k=2) + np.diag(c[1, 1:], k=1) + np.diag(c[2, :])\n    >>> np.allclose(C.conj().T @ C - A, np.zeros((5, 5)))\n    True\n\n    '
    if check_finite:
        ab = asarray_chkfinite(ab)
    else:
        ab = asarray(ab)
    (pbtrf,) = get_lapack_funcs(('pbtrf',), (ab,))
    (c, info) = pbtrf(ab, lower=lower, overwrite_ab=overwrite_ab)
    if info > 0:
        raise LinAlgError('%d-th leading minor not positive definite' % info)
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal pbtrf' % -info)
    return c

def cho_solve_banded(cb_and_lower, b, overwrite_b=False, check_finite=True):
    if False:
        while True:
            i = 10
    '\n    Solve the linear equations ``A x = b``, given the Cholesky factorization of\n    the banded Hermitian ``A``.\n\n    Parameters\n    ----------\n    (cb, lower) : tuple, (ndarray, bool)\n        `cb` is the Cholesky factorization of A, as given by cholesky_banded.\n        `lower` must be the same value that was given to cholesky_banded.\n    b : array_like\n        Right-hand side\n    overwrite_b : bool, optional\n        If True, the function will overwrite the values in `b`.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    x : array\n        The solution to the system A x = b\n\n    See Also\n    --------\n    cholesky_banded : Cholesky factorization of a banded matrix\n\n    Notes\n    -----\n\n    .. versionadded:: 0.8.0\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import cholesky_banded, cho_solve_banded\n    >>> Ab = np.array([[0, 0, 1j, 2, 3j], [0, -1, -2, 3, 4], [9, 8, 7, 6, 9]])\n    >>> A = np.diag(Ab[0,2:], k=2) + np.diag(Ab[1,1:], k=1)\n    >>> A = A + A.conj().T + np.diag(Ab[2, :])\n    >>> c = cholesky_banded(Ab)\n    >>> x = cho_solve_banded((c, False), np.ones(5))\n    >>> np.allclose(A @ x - np.ones(5), np.zeros(5))\n    True\n\n    '
    (cb, lower) = cb_and_lower
    if check_finite:
        cb = asarray_chkfinite(cb)
        b = asarray_chkfinite(b)
    else:
        cb = asarray(cb)
        b = asarray(b)
    if cb.shape[-1] != b.shape[0]:
        raise ValueError('shapes of cb and b are not compatible.')
    (pbtrs,) = get_lapack_funcs(('pbtrs',), (cb, b))
    (x, info) = pbtrs(cb, b, lower=lower, overwrite_b=overwrite_b)
    if info > 0:
        raise LinAlgError('%dth leading minor not positive definite' % info)
    if info < 0:
        raise ValueError('illegal value in %dth argument of internal pbtrs' % -info)
    return x