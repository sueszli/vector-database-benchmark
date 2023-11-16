from warnings import warn
import numpy as np
from numpy import atleast_2d, arange, zeros_like, imag, diag, iscomplexobj, tril, triu, argsort, empty_like
from scipy._lib._util import ComplexWarning
from ._decomp import _asarray_validated
from .lapack import get_lapack_funcs, _compute_lwork
__all__ = ['ldl']

def ldl(A, lower=True, hermitian=True, overwrite_a=False, check_finite=True):
    if False:
        return 10
    " Computes the LDLt or Bunch-Kaufman factorization of a symmetric/\n    hermitian matrix.\n\n    This function returns a block diagonal matrix D consisting blocks of size\n    at most 2x2 and also a possibly permuted unit lower triangular matrix\n    ``L`` such that the factorization ``A = L D L^H`` or ``A = L D L^T``\n    holds. If `lower` is False then (again possibly permuted) upper\n    triangular matrices are returned as outer factors.\n\n    The permutation array can be used to triangularize the outer factors\n    simply by a row shuffle, i.e., ``lu[perm, :]`` is an upper/lower\n    triangular matrix. This is also equivalent to multiplication with a\n    permutation matrix ``P.dot(lu)``, where ``P`` is a column-permuted\n    identity matrix ``I[:, perm]``.\n\n    Depending on the value of the boolean `lower`, only upper or lower\n    triangular part of the input array is referenced. Hence, a triangular\n    matrix on entry would give the same result as if the full matrix is\n    supplied.\n\n    Parameters\n    ----------\n    A : array_like\n        Square input array\n    lower : bool, optional\n        This switches between the lower and upper triangular outer factors of\n        the factorization. Lower triangular (``lower=True``) is the default.\n    hermitian : bool, optional\n        For complex-valued arrays, this defines whether ``A = A.conj().T`` or\n        ``A = A.T`` is assumed. For real-valued arrays, this switch has no\n        effect.\n    overwrite_a : bool, optional\n        Allow overwriting data in `A` (may enhance performance). The default\n        is False.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    lu : ndarray\n        The (possibly) permuted upper/lower triangular outer factor of the\n        factorization.\n    d : ndarray\n        The block diagonal multiplier of the factorization.\n    perm : ndarray\n        The row-permutation index array that brings lu into triangular form.\n\n    Raises\n    ------\n    ValueError\n        If input array is not square.\n    ComplexWarning\n        If a complex-valued array with nonzero imaginary parts on the\n        diagonal is given and hermitian is set to True.\n\n    See Also\n    --------\n    cholesky, lu\n\n    Notes\n    -----\n    This function uses ``?SYTRF`` routines for symmetric matrices and\n    ``?HETRF`` routines for Hermitian matrices from LAPACK. See [1]_ for\n    the algorithm details.\n\n    Depending on the `lower` keyword value, only lower or upper triangular\n    part of the input array is referenced. Moreover, this keyword also defines\n    the structure of the outer factors of the factorization.\n\n    .. versionadded:: 1.1.0\n\n    References\n    ----------\n    .. [1] J.R. Bunch, L. Kaufman, Some stable methods for calculating\n       inertia and solving symmetric linear systems, Math. Comput. Vol.31,\n       1977. :doi:`10.2307/2005787`\n\n    Examples\n    --------\n    Given an upper triangular array ``a`` that represents the full symmetric\n    array with its entries, obtain ``l``, 'd' and the permutation vector `perm`:\n\n    >>> import numpy as np\n    >>> from scipy.linalg import ldl\n    >>> a = np.array([[2, -1, 3], [0, 2, 0], [0, 0, 1]])\n    >>> lu, d, perm = ldl(a, lower=0) # Use the upper part\n    >>> lu\n    array([[ 0. ,  0. ,  1. ],\n           [ 0. ,  1. , -0.5],\n           [ 1. ,  1. ,  1.5]])\n    >>> d\n    array([[-5. ,  0. ,  0. ],\n           [ 0. ,  1.5,  0. ],\n           [ 0. ,  0. ,  2. ]])\n    >>> perm\n    array([2, 1, 0])\n    >>> lu[perm, :]\n    array([[ 1. ,  1. ,  1.5],\n           [ 0. ,  1. , -0.5],\n           [ 0. ,  0. ,  1. ]])\n    >>> lu.dot(d).dot(lu.T)\n    array([[ 2., -1.,  3.],\n           [-1.,  2.,  0.],\n           [ 3.,  0.,  1.]])\n\n    "
    a = atleast_2d(_asarray_validated(A, check_finite=check_finite))
    if a.shape[0] != a.shape[1]:
        raise ValueError('The input array "a" should be square.')
    if a.size == 0:
        return (empty_like(a), empty_like(a), np.array([], dtype=int))
    n = a.shape[0]
    r_or_c = complex if iscomplexobj(a) else float
    if r_or_c is complex and hermitian:
        (s, sl) = ('hetrf', 'hetrf_lwork')
        if np.any(imag(diag(a))):
            warn('scipy.linalg.ldl():\nThe imaginary parts of the diagonalare ignored. Use "hermitian=False" for factorization ofcomplex symmetric arrays.', ComplexWarning, stacklevel=2)
    else:
        (s, sl) = ('sytrf', 'sytrf_lwork')
    (solver, solver_lwork) = get_lapack_funcs((s, sl), (a,))
    lwork = _compute_lwork(solver_lwork, n, lower=lower)
    (ldu, piv, info) = solver(a, lwork=lwork, lower=lower, overwrite_a=overwrite_a)
    if info < 0:
        raise ValueError('{} exited with the internal error "illegal value in argument number {}". See LAPACK documentation for the error codes.'.format(s.upper(), -info))
    (swap_arr, pivot_arr) = _ldl_sanitize_ipiv(piv, lower=lower)
    (d, lu) = _ldl_get_d_and_l(ldu, pivot_arr, lower=lower, hermitian=hermitian)
    (lu, perm) = _ldl_construct_tri_factor(lu, swap_arr, pivot_arr, lower=lower)
    return (lu, d, perm)

def _ldl_sanitize_ipiv(a, lower=True):
    if False:
        print('Hello World!')
    "\n    This helper function takes the rather strangely encoded permutation array\n    returned by the LAPACK routines ?(HE/SY)TRF and converts it into\n    regularized permutation and diagonal pivot size format.\n\n    Since FORTRAN uses 1-indexing and LAPACK uses different start points for\n    upper and lower formats there are certain offsets in the indices used\n    below.\n\n    Let's assume a result where the matrix is 6x6 and there are two 2x2\n    and two 1x1 blocks reported by the routine. To ease the coding efforts,\n    we still populate a 6-sized array and fill zeros as the following ::\n\n        pivots = [2, 0, 2, 0, 1, 1]\n\n    This denotes a diagonal matrix of the form ::\n\n        [x x        ]\n        [x x        ]\n        [    x x    ]\n        [    x x    ]\n        [        x  ]\n        [          x]\n\n    In other words, we write 2 when the 2x2 block is first encountered and\n    automatically write 0 to the next entry and skip the next spin of the\n    loop. Thus, a separate counter or array appends to keep track of block\n    sizes are avoided. If needed, zeros can be filtered out later without\n    losing the block structure.\n\n    Parameters\n    ----------\n    a : ndarray\n        The permutation array ipiv returned by LAPACK\n    lower : bool, optional\n        The switch to select whether upper or lower triangle is chosen in\n        the LAPACK call.\n\n    Returns\n    -------\n    swap_ : ndarray\n        The array that defines the row/column swap operations. For example,\n        if row two is swapped with row four, the result is [0, 3, 2, 3].\n    pivots : ndarray\n        The array that defines the block diagonal structure as given above.\n\n    "
    n = a.size
    swap_ = arange(n)
    pivots = zeros_like(swap_, dtype=int)
    skip_2x2 = False
    (x, y, rs, re, ri) = (1, 0, 0, n, 1) if lower else (-1, -1, n - 1, -1, -1)
    for ind in range(rs, re, ri):
        if skip_2x2:
            skip_2x2 = False
            continue
        cur_val = a[ind]
        if cur_val > 0:
            if cur_val != ind + 1:
                swap_[ind] = swap_[cur_val - 1]
            pivots[ind] = 1
        elif cur_val < 0 and cur_val == a[ind + x]:
            if -cur_val != ind + 2:
                swap_[ind + x] = swap_[-cur_val - 1]
            pivots[ind + y] = 2
            skip_2x2 = True
        else:
            raise ValueError('While parsing the permutation array in "scipy.linalg.ldl", invalid entries found. The array syntax is invalid.')
    return (swap_, pivots)

def _ldl_get_d_and_l(ldu, pivs, lower=True, hermitian=True):
    if False:
        i = 10
        return i + 15
    '\n    Helper function to extract the diagonal and triangular matrices for\n    LDL.T factorization.\n\n    Parameters\n    ----------\n    ldu : ndarray\n        The compact output returned by the LAPACK routing\n    pivs : ndarray\n        The sanitized array of {0, 1, 2} denoting the sizes of the pivots. For\n        every 2 there is a succeeding 0.\n    lower : bool, optional\n        If set to False, upper triangular part is considered.\n    hermitian : bool, optional\n        If set to False a symmetric complex array is assumed.\n\n    Returns\n    -------\n    d : ndarray\n        The block diagonal matrix.\n    lu : ndarray\n        The upper/lower triangular matrix\n    '
    is_c = iscomplexobj(ldu)
    d = diag(diag(ldu))
    n = d.shape[0]
    blk_i = 0
    (x, y) = (1, 0) if lower else (0, 1)
    lu = tril(ldu, -1) if lower else triu(ldu, 1)
    diag_inds = arange(n)
    lu[diag_inds, diag_inds] = 1
    for blk in pivs[pivs != 0]:
        inc = blk_i + blk
        if blk == 2:
            d[blk_i + x, blk_i + y] = ldu[blk_i + x, blk_i + y]
            if is_c and hermitian:
                d[blk_i + y, blk_i + x] = ldu[blk_i + x, blk_i + y].conj()
            else:
                d[blk_i + y, blk_i + x] = ldu[blk_i + x, blk_i + y]
            lu[blk_i + x, blk_i + y] = 0.0
        blk_i = inc
    return (d, lu)

def _ldl_construct_tri_factor(lu, swap_vec, pivs, lower=True):
    if False:
        return 10
    '\n    Helper function to construct explicit outer factors of LDL factorization.\n\n    If lower is True the permuted factors are multiplied as L(1)*L(2)*...*L(k).\n    Otherwise, the permuted factors are multiplied as L(k)*...*L(2)*L(1). See\n    LAPACK documentation for more details.\n\n    Parameters\n    ----------\n    lu : ndarray\n        The triangular array that is extracted from LAPACK routine call with\n        ones on the diagonals.\n    swap_vec : ndarray\n        The array that defines the row swapping indices. If the kth entry is m\n        then rows k,m are swapped. Notice that the mth entry is not necessarily\n        k to avoid undoing the swapping.\n    pivs : ndarray\n        The array that defines the block diagonal structure returned by\n        _ldl_sanitize_ipiv().\n    lower : bool, optional\n        The boolean to switch between lower and upper triangular structure.\n\n    Returns\n    -------\n    lu : ndarray\n        The square outer factor which satisfies the L * D * L.T = A\n    perm : ndarray\n        The permutation vector that brings the lu to the triangular form\n\n    Notes\n    -----\n    Note that the original argument "lu" is overwritten.\n\n    '
    n = lu.shape[0]
    perm = arange(n)
    (rs, re, ri) = (n - 1, -1, -1) if lower else (0, n, 1)
    for ind in range(rs, re, ri):
        s_ind = swap_vec[ind]
        if s_ind != ind:
            col_s = ind if lower else 0
            col_e = n if lower else ind + 1
            if pivs[ind] == (0 if lower else 2):
                col_s += -1 if lower else 0
                col_e += 0 if lower else 1
            lu[[s_ind, ind], col_s:col_e] = lu[[ind, s_ind], col_s:col_e]
            perm[[s_ind, ind]] = perm[[ind, s_ind]]
    return (lu, argsort(perm))