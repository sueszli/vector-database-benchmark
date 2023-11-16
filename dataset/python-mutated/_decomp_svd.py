"""SVD decomposition functions."""
import numpy
from numpy import zeros, r_, diag, dot, arccos, arcsin, where, clip
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs, _compute_lwork
from ._decomp import _asarray_validated
__all__ = ['svd', 'svdvals', 'diagsvd', 'orth', 'subspace_angles', 'null_space']

def svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Singular Value Decomposition.\n\n    Factorizes the matrix `a` into two unitary matrices ``U`` and ``Vh``, and\n    a 1-D array ``s`` of singular values (real, non-negative) such that\n    ``a == U @ S @ Vh``, where ``S`` is a suitably shaped matrix of zeros with\n    main diagonal ``s``.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        Matrix to decompose.\n    full_matrices : bool, optional\n        If True (default), `U` and `Vh` are of shape ``(M, M)``, ``(N, N)``.\n        If False, the shapes are ``(M, K)`` and ``(K, N)``, where\n        ``K = min(M, N)``.\n    compute_uv : bool, optional\n        Whether to compute also ``U`` and ``Vh`` in addition to ``s``.\n        Default is True.\n    overwrite_a : bool, optional\n        Whether to overwrite `a`; may improve performance.\n        Default is False.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n    lapack_driver : {'gesdd', 'gesvd'}, optional\n        Whether to use the more efficient divide-and-conquer approach\n        (``'gesdd'``) or general rectangular approach (``'gesvd'``)\n        to compute the SVD. MATLAB and Octave use the ``'gesvd'`` approach.\n        Default is ``'gesdd'``.\n\n        .. versionadded:: 0.18\n\n    Returns\n    -------\n    U : ndarray\n        Unitary matrix having left singular vectors as columns.\n        Of shape ``(M, M)`` or ``(M, K)``, depending on `full_matrices`.\n    s : ndarray\n        The singular values, sorted in non-increasing order.\n        Of shape (K,), with ``K = min(M, N)``.\n    Vh : ndarray\n        Unitary matrix having right singular vectors as rows.\n        Of shape ``(N, N)`` or ``(K, N)`` depending on `full_matrices`.\n\n    For ``compute_uv=False``, only ``s`` is returned.\n\n    Raises\n    ------\n    LinAlgError\n        If SVD computation does not converge.\n\n    See Also\n    --------\n    svdvals : Compute singular values of a matrix.\n    diagsvd : Construct the Sigma matrix, given the vector s.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import linalg\n    >>> rng = np.random.default_rng()\n    >>> m, n = 9, 6\n    >>> a = rng.standard_normal((m, n)) + 1.j*rng.standard_normal((m, n))\n    >>> U, s, Vh = linalg.svd(a)\n    >>> U.shape,  s.shape, Vh.shape\n    ((9, 9), (6,), (6, 6))\n\n    Reconstruct the original matrix from the decomposition:\n\n    >>> sigma = np.zeros((m, n))\n    >>> for i in range(min(m, n)):\n    ...     sigma[i, i] = s[i]\n    >>> a1 = np.dot(U, np.dot(sigma, Vh))\n    >>> np.allclose(a, a1)\n    True\n\n    Alternatively, use ``full_matrices=False`` (notice that the shape of\n    ``U`` is then ``(m, n)`` instead of ``(m, m)``):\n\n    >>> U, s, Vh = linalg.svd(a, full_matrices=False)\n    >>> U.shape, s.shape, Vh.shape\n    ((9, 6), (6,), (6, 6))\n    >>> S = np.diag(s)\n    >>> np.allclose(a, np.dot(U, np.dot(S, Vh)))\n    True\n\n    >>> s2 = linalg.svd(a, compute_uv=False)\n    >>> np.allclose(s, s2)\n    True\n\n    "
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2:
        raise ValueError('expected matrix')
    (m, n) = a1.shape
    overwrite_a = overwrite_a or _datacopied(a1, a)
    if not isinstance(lapack_driver, str):
        raise TypeError('lapack_driver must be a string')
    if lapack_driver not in ('gesdd', 'gesvd'):
        raise ValueError('lapack_driver must be "gesdd" or "gesvd", not "%s"' % (lapack_driver,))
    funcs = (lapack_driver, lapack_driver + '_lwork')
    (gesXd, gesXd_lwork) = get_lapack_funcs(funcs, (a1,), ilp64='preferred')
    lwork = _compute_lwork(gesXd_lwork, a1.shape[0], a1.shape[1], compute_uv=compute_uv, full_matrices=full_matrices)
    (u, s, v, info) = gesXd(a1, compute_uv=compute_uv, lwork=lwork, full_matrices=full_matrices, overwrite_a=overwrite_a)
    if info > 0:
        raise LinAlgError('SVD did not converge')
    if info < 0:
        raise ValueError('illegal value in %dth argument of internal gesdd' % -info)
    if compute_uv:
        return (u, s, v)
    else:
        return s

def svdvals(a, overwrite_a=False, check_finite=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute singular values of a matrix.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        Matrix to decompose.\n    overwrite_a : bool, optional\n        Whether to overwrite `a`; may improve performance.\n        Default is False.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    s : (min(M, N),) ndarray\n        The singular values, sorted in decreasing order.\n\n    Raises\n    ------\n    LinAlgError\n        If SVD computation does not converge.\n\n    See Also\n    --------\n    svd : Compute the full singular value decomposition of a matrix.\n    diagsvd : Construct the Sigma matrix, given the vector s.\n\n    Notes\n    -----\n    ``svdvals(a)`` only differs from ``svd(a, compute_uv=False)`` by its\n    handling of the edge case of empty ``a``, where it returns an\n    empty sequence:\n\n    >>> import numpy as np\n    >>> a = np.empty((0, 2))\n    >>> from scipy.linalg import svdvals\n    >>> svdvals(a)\n    array([], dtype=float64)\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import svdvals\n    >>> m = np.array([[1.0, 0.0],\n    ...               [2.0, 3.0],\n    ...               [1.0, 1.0],\n    ...               [0.0, 2.0],\n    ...               [1.0, 0.0]])\n    >>> svdvals(m)\n    array([ 4.28091555,  1.63516424])\n\n    We can verify the maximum singular value of `m` by computing the maximum\n    length of `m.dot(u)` over all the unit vectors `u` in the (x,y) plane.\n    We approximate "all" the unit vectors with a large sample. Because\n    of linearity, we only need the unit vectors with angles in [0, pi].\n\n    >>> t = np.linspace(0, np.pi, 2000)\n    >>> u = np.array([np.cos(t), np.sin(t)])\n    >>> np.linalg.norm(m.dot(u), axis=0).max()\n    4.2809152422538475\n\n    `p` is a projection matrix with rank 1. With exact arithmetic,\n    its singular values would be [1, 0, 0, 0].\n\n    >>> v = np.array([0.1, 0.3, 0.9, 0.3])\n    >>> p = np.outer(v, v)\n    >>> svdvals(p)\n    array([  1.00000000e+00,   2.02021698e-17,   1.56692500e-17,\n             8.15115104e-34])\n\n    The singular values of an orthogonal matrix are all 1. Here, we\n    create a random orthogonal matrix by using the `rvs()` method of\n    `scipy.stats.ortho_group`.\n\n    >>> from scipy.stats import ortho_group\n    >>> orth = ortho_group.rvs(4)\n    >>> svdvals(orth)\n    array([ 1.,  1.,  1.,  1.])\n\n    '
    a = _asarray_validated(a, check_finite=check_finite)
    if a.size:
        return svd(a, compute_uv=0, overwrite_a=overwrite_a, check_finite=False)
    elif len(a.shape) != 2:
        raise ValueError('expected matrix')
    else:
        return numpy.empty(0)

def diagsvd(s, M, N):
    if False:
        print('Hello World!')
    '\n    Construct the sigma matrix in SVD from singular values and size M, N.\n\n    Parameters\n    ----------\n    s : (M,) or (N,) array_like\n        Singular values\n    M : int\n        Size of the matrix whose singular values are `s`.\n    N : int\n        Size of the matrix whose singular values are `s`.\n\n    Returns\n    -------\n    S : (M, N) ndarray\n        The S-matrix in the singular value decomposition\n\n    See Also\n    --------\n    svd : Singular value decomposition of a matrix\n    svdvals : Compute singular values of a matrix.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import diagsvd\n    >>> vals = np.array([1, 2, 3])  # The array representing the computed svd\n    >>> diagsvd(vals, 3, 4)\n    array([[1, 0, 0, 0],\n           [0, 2, 0, 0],\n           [0, 0, 3, 0]])\n    >>> diagsvd(vals, 4, 3)\n    array([[1, 0, 0],\n           [0, 2, 0],\n           [0, 0, 3],\n           [0, 0, 0]])\n\n    '
    part = diag(s)
    typ = part.dtype.char
    MorN = len(s)
    if MorN == M:
        return numpy.hstack((part, zeros((M, N - M), dtype=typ)))
    elif MorN == N:
        return r_[part, zeros((M - N, N), dtype=typ)]
    else:
        raise ValueError('Length of s must be M or N.')

def orth(A, rcond=None):
    if False:
        return 10
    '\n    Construct an orthonormal basis for the range of A using SVD\n\n    Parameters\n    ----------\n    A : (M, N) array_like\n        Input array\n    rcond : float, optional\n        Relative condition number. Singular values ``s`` smaller than\n        ``rcond * max(s)`` are considered zero.\n        Default: floating point eps * max(M,N).\n\n    Returns\n    -------\n    Q : (M, K) ndarray\n        Orthonormal basis for the range of A.\n        K = effective rank of A, as determined by rcond\n\n    See Also\n    --------\n    svd : Singular value decomposition of a matrix\n    null_space : Matrix null space\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import orth\n    >>> A = np.array([[2, 0, 0], [0, 5, 0]])  # rank 2 array\n    >>> orth(A)\n    array([[0., 1.],\n           [1., 0.]])\n    >>> orth(A.T)\n    array([[0., 1.],\n           [1., 0.],\n           [0., 0.]])\n\n    '
    (u, s, vh) = svd(A, full_matrices=False)
    (M, N) = (u.shape[0], vh.shape[1])
    if rcond is None:
        rcond = numpy.finfo(s.dtype).eps * max(M, N)
    tol = numpy.amax(s) * rcond
    num = numpy.sum(s > tol, dtype=int)
    Q = u[:, :num]
    return Q

def null_space(A, rcond=None):
    if False:
        while True:
            i = 10
    '\n    Construct an orthonormal basis for the null space of A using SVD\n\n    Parameters\n    ----------\n    A : (M, N) array_like\n        Input array\n    rcond : float, optional\n        Relative condition number. Singular values ``s`` smaller than\n        ``rcond * max(s)`` are considered zero.\n        Default: floating point eps * max(M,N).\n\n    Returns\n    -------\n    Z : (N, K) ndarray\n        Orthonormal basis for the null space of A.\n        K = dimension of effective null space, as determined by rcond\n\n    See Also\n    --------\n    svd : Singular value decomposition of a matrix\n    orth : Matrix range\n\n    Examples\n    --------\n    1-D null space:\n\n    >>> import numpy as np\n    >>> from scipy.linalg import null_space\n    >>> A = np.array([[1, 1], [1, 1]])\n    >>> ns = null_space(A)\n    >>> ns * np.copysign(1, ns[0,0])  # Remove the sign ambiguity of the vector\n    array([[ 0.70710678],\n           [-0.70710678]])\n\n    2-D null space:\n\n    >>> from numpy.random import default_rng\n    >>> rng = default_rng()\n    >>> B = rng.random((3, 5))\n    >>> Z = null_space(B)\n    >>> Z.shape\n    (5, 2)\n    >>> np.allclose(B.dot(Z), 0)\n    True\n\n    The basis vectors are orthonormal (up to rounding error):\n\n    >>> Z.T.dot(Z)\n    array([[  1.00000000e+00,   6.92087741e-17],\n           [  6.92087741e-17,   1.00000000e+00]])\n\n    '
    (u, s, vh) = svd(A, full_matrices=True)
    (M, N) = (u.shape[0], vh.shape[1])
    if rcond is None:
        rcond = numpy.finfo(s.dtype).eps * max(M, N)
    tol = numpy.amax(s) * rcond
    num = numpy.sum(s > tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q

def subspace_angles(A, B):
    if False:
        i = 10
        return i + 15
    '\n    Compute the subspace angles between two matrices.\n\n    Parameters\n    ----------\n    A : (M, N) array_like\n        The first input array.\n    B : (M, K) array_like\n        The second input array.\n\n    Returns\n    -------\n    angles : ndarray, shape (min(N, K),)\n        The subspace angles between the column spaces of `A` and `B` in\n        descending order.\n\n    See Also\n    --------\n    orth\n    svd\n\n    Notes\n    -----\n    This computes the subspace angles according to the formula\n    provided in [1]_. For equivalence with MATLAB and Octave behavior,\n    use ``angles[0]``.\n\n    .. versionadded:: 1.0\n\n    References\n    ----------\n    .. [1] Knyazev A, Argentati M (2002) Principal Angles between Subspaces\n           in an A-Based Scalar Product: Algorithms and Perturbation\n           Estimates. SIAM J. Sci. Comput. 23:2008-2040.\n\n    Examples\n    --------\n    An Hadamard matrix, which has orthogonal columns, so we expect that\n    the suspace angle to be :math:`\\frac{\\pi}{2}`:\n\n    >>> import numpy as np\n    >>> from scipy.linalg import hadamard, subspace_angles\n    >>> rng = np.random.default_rng()\n    >>> H = hadamard(4)\n    >>> print(H)\n    [[ 1  1  1  1]\n     [ 1 -1  1 -1]\n     [ 1  1 -1 -1]\n     [ 1 -1 -1  1]]\n    >>> np.rad2deg(subspace_angles(H[:, :2], H[:, 2:]))\n    array([ 90.,  90.])\n\n    And the subspace angle of a matrix to itself should be zero:\n\n    >>> subspace_angles(H[:, :2], H[:, :2]) <= 2 * np.finfo(float).eps\n    array([ True,  True], dtype=bool)\n\n    The angles between non-orthogonal subspaces are in between these extremes:\n\n    >>> x = rng.standard_normal((4, 3))\n    >>> np.rad2deg(subspace_angles(x[:, :2], x[:, [2]]))\n    array([ 55.832])  # random\n    '
    A = _asarray_validated(A, check_finite=True)
    if len(A.shape) != 2:
        raise ValueError(f'expected 2D array, got shape {A.shape}')
    QA = orth(A)
    del A
    B = _asarray_validated(B, check_finite=True)
    if len(B.shape) != 2:
        raise ValueError(f'expected 2D array, got shape {B.shape}')
    if len(B) != len(QA):
        raise ValueError('A and B must have the same number of rows, got {} and {}'.format(QA.shape[0], B.shape[0]))
    QB = orth(B)
    del B
    QA_H_QB = dot(QA.T.conj(), QB)
    sigma = svdvals(QA_H_QB)
    if QA.shape[1] >= QB.shape[1]:
        B = QB - dot(QA, QA_H_QB)
    else:
        B = QA - dot(QB, QA_H_QB.T.conj())
    del QA, QB, QA_H_QB
    mask = sigma ** 2 >= 0.5
    if mask.any():
        mu_arcsin = arcsin(clip(svdvals(B, overwrite_a=True), -1.0, 1.0))
    else:
        mu_arcsin = 0.0
    theta = where(mask, mu_arcsin, arccos(clip(sigma[::-1], -1.0, 1.0)))
    return theta