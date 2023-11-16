"""
The :mod:`sklearn.utils.extmath` module includes utilities to perform
optimal mathematical operations in scikit-learn that are not available in SciPy.
"""
import warnings
from functools import partial
from numbers import Integral
import numpy as np
from scipy import linalg, sparse
from ..utils import deprecated
from ..utils._param_validation import Interval, StrOptions, validate_params
from . import check_random_state
from ._array_api import _is_numpy_namespace, device, get_namespace
from .sparsefuncs_fast import csr_row_norms
from .validation import check_array

def squared_norm(x):
    if False:
        print('Hello World!')
    'Squared Euclidean or Frobenius norm of x.\n\n    Faster than norm(x) ** 2.\n\n    Parameters\n    ----------\n    x : array-like\n        The input array which could be either be a vector or a 2 dimensional array.\n\n    Returns\n    -------\n    float\n        The Euclidean norm when x is a vector, the Frobenius norm when x\n        is a matrix (2-d array).\n    '
    x = np.ravel(x, order='K')
    if np.issubdtype(x.dtype, np.integer):
        warnings.warn('Array type is integer, np.dot may overflow. Data should be float type to avoid this issue', UserWarning)
    return np.dot(x, x)

def row_norms(X, squared=False):
    if False:
        for i in range(10):
            print('nop')
    'Row-wise (squared) Euclidean norm of X.\n\n    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse\n    matrices and does not create an X.shape-sized temporary.\n\n    Performs no input validation.\n\n    Parameters\n    ----------\n    X : array-like\n        The input array.\n    squared : bool, default=False\n        If True, return squared norms.\n\n    Returns\n    -------\n    array-like\n        The row-wise (squared) Euclidean norm of X.\n    '
    if sparse.issparse(X):
        X = X.tocsr()
        norms = csr_row_norms(X)
        if not squared:
            norms = np.sqrt(norms)
    else:
        (xp, _) = get_namespace(X)
        if _is_numpy_namespace(xp):
            X = np.asarray(X)
            norms = np.einsum('ij,ij->i', X, X)
            norms = xp.asarray(norms)
        else:
            norms = xp.sum(xp.multiply(X, X), axis=1)
        if not squared:
            norms = xp.sqrt(norms)
    return norms

def fast_logdet(A):
    if False:
        i = 10
        return i + 15
    'Compute logarithm of determinant of a square matrix.\n\n    The (natural) logarithm of the determinant of a square matrix\n    is returned if det(A) is non-negative and well defined.\n    If the determinant is zero or negative returns -Inf.\n\n    Equivalent to : np.log(np.det(A)) but more robust.\n\n    Parameters\n    ----------\n    A : array_like of shape (n, n)\n        The square matrix.\n\n    Returns\n    -------\n    logdet : float\n        When det(A) is strictly positive, log(det(A)) is returned.\n        When det(A) is non-positive or not defined, then -inf is returned.\n\n    See Also\n    --------\n    numpy.linalg.slogdet : Compute the sign and (natural) logarithm of the determinant\n        of an array.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from sklearn.utils.extmath import fast_logdet\n    >>> a = np.array([[5, 1], [2, 8]])\n    >>> fast_logdet(a)\n    3.6375861597263857\n    '
    (xp, _) = get_namespace(A)
    (sign, ld) = xp.linalg.slogdet(A)
    if not sign > 0:
        return -xp.inf
    return ld

def density(w, **kwargs):
    if False:
        i = 10
        return i + 15
    'Compute density of a sparse vector.\n\n    Parameters\n    ----------\n    w : array-like\n        The sparse vector.\n    **kwargs : keyword arguments\n        Ignored.\n\n        .. deprecated:: 1.2\n            ``**kwargs`` were deprecated in version 1.2 and will be removed in\n            1.4.\n\n    Returns\n    -------\n    float\n        The density of w, between 0 and 1.\n    '
    if kwargs:
        warnings.warn('Additional keyword arguments are deprecated in version 1.2 and will be removed in version 1.4.', FutureWarning)
    if hasattr(w, 'toarray'):
        d = float(w.nnz) / (w.shape[0] * w.shape[1])
    else:
        d = 0 if w is None else float((w != 0).sum()) / w.size
    return d

def safe_sparse_dot(a, b, *, dense_output=False):
    if False:
        for i in range(10):
            print('nop')
    'Dot product that handle the sparse matrix case correctly.\n\n    Parameters\n    ----------\n    a : {ndarray, sparse matrix}\n    b : {ndarray, sparse matrix}\n    dense_output : bool, default=False\n        When False, ``a`` and ``b`` both being sparse will yield sparse output.\n        When True, output will always be a dense array.\n\n    Returns\n    -------\n    dot_product : {ndarray, sparse matrix}\n        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.\n    '
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b
    if sparse.issparse(a) and sparse.issparse(b) and dense_output and hasattr(ret, 'toarray'):
        return ret.toarray()
    return ret

def randomized_range_finder(A, *, size, n_iter, power_iteration_normalizer='auto', random_state=None):
    if False:
        while True:
            i = 10
    'Compute an orthonormal matrix whose range approximates the range of A.\n\n    Parameters\n    ----------\n    A : 2D array\n        The input data matrix.\n\n    size : int\n        Size of the return array.\n\n    n_iter : int\n        Number of power iterations used to stabilize the result.\n\n    power_iteration_normalizer : {\'auto\', \'QR\', \'LU\', \'none\'}, default=\'auto\'\n        Whether the power iterations are normalized with step-by-step\n        QR factorization (the slowest but most accurate), \'none\'\n        (the fastest but numerically unstable when `n_iter` is large, e.g.\n        typically 5 or larger), or \'LU\' factorization (numerically stable\n        but can lose slightly in accuracy). The \'auto\' mode applies no\n        normalization if `n_iter` <= 2 and switches to LU otherwise.\n\n        .. versionadded:: 0.18\n\n    random_state : int, RandomState instance or None, default=None\n        The seed of the pseudo random number generator to use when shuffling\n        the data, i.e. getting the random vectors to initialize the algorithm.\n        Pass an int for reproducible results across multiple function calls.\n        See :term:`Glossary <random_state>`.\n\n    Returns\n    -------\n    Q : ndarray\n        A (size x size) projection matrix, the range of which\n        approximates well the range of the input matrix A.\n\n    Notes\n    -----\n\n    Follows Algorithm 4.3 of\n    :arxiv:`"Finding structure with randomness:\n    Stochastic algorithms for constructing approximate matrix decompositions"\n    <0909.4061>`\n    Halko, et al. (2009)\n\n    An implementation of a randomized algorithm for principal component\n    analysis\n    A. Szlam et al. 2014\n    '
    (xp, is_array_api_compliant) = get_namespace(A)
    random_state = check_random_state(random_state)
    Q = xp.asarray(random_state.normal(size=(A.shape[1], size)))
    if hasattr(A, 'dtype') and xp.isdtype(A.dtype, kind='real floating'):
        Q = xp.astype(Q, A.dtype, copy=False)
    if is_array_api_compliant:
        Q = xp.asarray(Q, device=device(A))
    if power_iteration_normalizer == 'auto':
        if n_iter <= 2:
            power_iteration_normalizer = 'none'
        elif is_array_api_compliant:
            warnings.warn("Array API does not support LU factorization, falling back to QR instead. Set `power_iteration_normalizer='QR'` explicitly to silence this warning.")
            power_iteration_normalizer = 'QR'
        else:
            power_iteration_normalizer = 'LU'
    elif power_iteration_normalizer == 'LU' and is_array_api_compliant:
        raise ValueError("Array API does not support LU factorization. Set `power_iteration_normalizer='QR'` instead.")
    if is_array_api_compliant:
        qr_normalizer = partial(xp.linalg.qr, mode='reduced')
    else:
        qr_normalizer = partial(linalg.qr, mode='economic')
    if power_iteration_normalizer == 'QR':
        normalizer = qr_normalizer
    elif power_iteration_normalizer == 'LU':
        normalizer = partial(linalg.lu, permute_l=True)
    else:
        normalizer = lambda x: (x, None)
    for _ in range(n_iter):
        (Q, _) = normalizer(A @ Q)
        (Q, _) = normalizer(A.T @ Q)
    (Q, _) = qr_normalizer(A @ Q)
    return Q

@validate_params({'M': [np.ndarray, 'sparse matrix'], 'n_components': [Interval(Integral, 1, None, closed='left')], 'n_oversamples': [Interval(Integral, 0, None, closed='left')], 'n_iter': [Interval(Integral, 0, None, closed='left'), StrOptions({'auto'})], 'power_iteration_normalizer': [StrOptions({'auto', 'QR', 'LU', 'none'})], 'transpose': ['boolean', StrOptions({'auto'})], 'flip_sign': ['boolean'], 'random_state': ['random_state'], 'svd_lapack_driver': [StrOptions({'gesdd', 'gesvd'})]}, prefer_skip_nested_validation=True)
def randomized_svd(M, n_components, *, n_oversamples=10, n_iter='auto', power_iteration_normalizer='auto', transpose='auto', flip_sign=True, random_state=None, svd_lapack_driver='gesdd'):
    if False:
        while True:
            i = 10
    'Compute a truncated randomized SVD.\n\n    This method solves the fixed-rank approximation problem described in [1]_\n    (problem (1.5), p5).\n\n    Parameters\n    ----------\n    M : {ndarray, sparse matrix}\n        Matrix to decompose.\n\n    n_components : int\n        Number of singular values and vectors to extract.\n\n    n_oversamples : int, default=10\n        Additional number of random vectors to sample the range of `M` so as\n        to ensure proper conditioning. The total number of random vectors\n        used to find the range of `M` is `n_components + n_oversamples`. Smaller\n        number can improve speed but can negatively impact the quality of\n        approximation of singular vectors and singular values. Users might wish\n        to increase this parameter up to `2*k - n_components` where k is the\n        effective rank, for large matrices, noisy problems, matrices with\n        slowly decaying spectrums, or to increase precision accuracy. See [1]_\n        (pages 5, 23 and 26).\n\n    n_iter : int or \'auto\', default=\'auto\'\n        Number of power iterations. It can be used to deal with very noisy\n        problems. When \'auto\', it is set to 4, unless `n_components` is small\n        (< .1 * min(X.shape)) in which case `n_iter` is set to 7.\n        This improves precision with few components. Note that in general\n        users should rather increase `n_oversamples` before increasing `n_iter`\n        as the principle of the randomized method is to avoid usage of these\n        more costly power iterations steps. When `n_components` is equal\n        or greater to the effective matrix rank and the spectrum does not\n        present a slow decay, `n_iter=0` or `1` should even work fine in theory\n        (see [1]_ page 9).\n\n        .. versionchanged:: 0.18\n\n    power_iteration_normalizer : {\'auto\', \'QR\', \'LU\', \'none\'}, default=\'auto\'\n        Whether the power iterations are normalized with step-by-step\n        QR factorization (the slowest but most accurate), \'none\'\n        (the fastest but numerically unstable when `n_iter` is large, e.g.\n        typically 5 or larger), or \'LU\' factorization (numerically stable\n        but can lose slightly in accuracy). The \'auto\' mode applies no\n        normalization if `n_iter` <= 2 and switches to LU otherwise.\n\n        .. versionadded:: 0.18\n\n    transpose : bool or \'auto\', default=\'auto\'\n        Whether the algorithm should be applied to M.T instead of M. The\n        result should approximately be the same. The \'auto\' mode will\n        trigger the transposition if M.shape[1] > M.shape[0] since this\n        implementation of randomized SVD tend to be a little faster in that\n        case.\n\n        .. versionchanged:: 0.18\n\n    flip_sign : bool, default=True\n        The output of a singular value decomposition is only unique up to a\n        permutation of the signs of the singular vectors. If `flip_sign` is\n        set to `True`, the sign ambiguity is resolved by making the largest\n        loadings for each component in the left singular vectors positive.\n\n    random_state : int, RandomState instance or None, default=\'warn\'\n        The seed of the pseudo random number generator to use when\n        shuffling the data, i.e. getting the random vectors to initialize\n        the algorithm. Pass an int for reproducible results across multiple\n        function calls. See :term:`Glossary <random_state>`.\n\n        .. versionchanged:: 1.2\n            The default value changed from 0 to None.\n\n    svd_lapack_driver : {"gesdd", "gesvd"}, default="gesdd"\n        Whether to use the more efficient divide-and-conquer approach\n        (`"gesdd"`) or more general rectangular approach (`"gesvd"`) to compute\n        the SVD of the matrix B, which is the projection of M into a low\n        dimensional subspace, as described in [1]_.\n\n        .. versionadded:: 1.2\n\n    Returns\n    -------\n    u : ndarray of shape (n_samples, n_components)\n        Unitary matrix having left singular vectors with signs flipped as columns.\n    s : ndarray of shape (n_components,)\n        The singular values, sorted in non-increasing order.\n    vh : ndarray of shape (n_components, n_features)\n        Unitary matrix having right singular vectors with signs flipped as rows.\n\n    Notes\n    -----\n    This algorithm finds a (usually very good) approximate truncated\n    singular value decomposition using randomization to speed up the\n    computations. It is particularly fast on large matrices on which\n    you wish to extract only a small number of components. In order to\n    obtain further speed up, `n_iter` can be set <=2 (at the cost of\n    loss of precision). To increase the precision it is recommended to\n    increase `n_oversamples`, up to `2*k-n_components` where k is the\n    effective rank. Usually, `n_components` is chosen to be greater than k\n    so increasing `n_oversamples` up to `n_components` should be enough.\n\n    References\n    ----------\n    .. [1] :arxiv:`"Finding structure with randomness:\n      Stochastic algorithms for constructing approximate matrix decompositions"\n      <0909.4061>`\n      Halko, et al. (2009)\n\n    .. [2] A randomized algorithm for the decomposition of matrices\n      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert\n\n    .. [3] An implementation of a randomized algorithm for principal component\n      analysis A. Szlam et al. 2014\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from sklearn.utils.extmath import randomized_svd\n    >>> a = np.array([[1, 2, 3, 5],\n    ...               [3, 4, 5, 6],\n    ...               [7, 8, 9, 10]])\n    >>> U, s, Vh = randomized_svd(a, n_components=2, random_state=0)\n    >>> U.shape, s.shape, Vh.shape\n    ((3, 2), (2,), (2, 4))\n    '
    if sparse.issparse(M) and M.format in ('lil', 'dok'):
        warnings.warn('Calculating SVD of a {} is expensive. csr_matrix is more efficient.'.format(type(M).__name__), sparse.SparseEfficiencyWarning)
    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    (n_samples, n_features) = M.shape
    if n_iter == 'auto':
        n_iter = 7 if n_components < 0.1 * min(M.shape) else 4
    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        M = M.T
    Q = randomized_range_finder(M, size=n_random, n_iter=n_iter, power_iteration_normalizer=power_iteration_normalizer, random_state=random_state)
    B = Q.T @ M
    (xp, is_array_api_compliant) = get_namespace(B)
    if is_array_api_compliant:
        (Uhat, s, Vt) = xp.linalg.svd(B, full_matrices=False)
    else:
        (Uhat, s, Vt) = linalg.svd(B, full_matrices=False, lapack_driver=svd_lapack_driver)
    del B
    U = Q @ Uhat
    if flip_sign:
        if not transpose:
            (U, Vt) = svd_flip(U, Vt)
        else:
            (U, Vt) = svd_flip(U, Vt, u_based_decision=False)
    if transpose:
        return (Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T)
    else:
        return (U[:, :n_components], s[:n_components], Vt[:n_components, :])

def _randomized_eigsh(M, n_components, *, n_oversamples=10, n_iter='auto', power_iteration_normalizer='auto', selection='module', random_state=None):
    if False:
        return 10
    'Computes a truncated eigendecomposition using randomized methods\n\n    This method solves the fixed-rank approximation problem described in the\n    Halko et al paper.\n\n    The choice of which components to select can be tuned with the `selection`\n    parameter.\n\n    .. versionadded:: 0.24\n\n    Parameters\n    ----------\n    M : ndarray or sparse matrix\n        Matrix to decompose, it should be real symmetric square or complex\n        hermitian\n\n    n_components : int\n        Number of eigenvalues and vectors to extract.\n\n    n_oversamples : int, default=10\n        Additional number of random vectors to sample the range of M so as\n        to ensure proper conditioning. The total number of random vectors\n        used to find the range of M is n_components + n_oversamples. Smaller\n        number can improve speed but can negatively impact the quality of\n        approximation of eigenvectors and eigenvalues. Users might wish\n        to increase this parameter up to `2*k - n_components` where k is the\n        effective rank, for large matrices, noisy problems, matrices with\n        slowly decaying spectrums, or to increase precision accuracy. See Halko\n        et al (pages 5, 23 and 26).\n\n    n_iter : int or \'auto\', default=\'auto\'\n        Number of power iterations. It can be used to deal with very noisy\n        problems. When \'auto\', it is set to 4, unless `n_components` is small\n        (< .1 * min(X.shape)) in which case `n_iter` is set to 7.\n        This improves precision with few components. Note that in general\n        users should rather increase `n_oversamples` before increasing `n_iter`\n        as the principle of the randomized method is to avoid usage of these\n        more costly power iterations steps. When `n_components` is equal\n        or greater to the effective matrix rank and the spectrum does not\n        present a slow decay, `n_iter=0` or `1` should even work fine in theory\n        (see Halko et al paper, page 9).\n\n    power_iteration_normalizer : {\'auto\', \'QR\', \'LU\', \'none\'}, default=\'auto\'\n        Whether the power iterations are normalized with step-by-step\n        QR factorization (the slowest but most accurate), \'none\'\n        (the fastest but numerically unstable when `n_iter` is large, e.g.\n        typically 5 or larger), or \'LU\' factorization (numerically stable\n        but can lose slightly in accuracy). The \'auto\' mode applies no\n        normalization if `n_iter` <= 2 and switches to LU otherwise.\n\n    selection : {\'value\', \'module\'}, default=\'module\'\n        Strategy used to select the n components. When `selection` is `\'value\'`\n        (not yet implemented, will become the default when implemented), the\n        components corresponding to the n largest eigenvalues are returned.\n        When `selection` is `\'module\'`, the components corresponding to the n\n        eigenvalues with largest modules are returned.\n\n    random_state : int, RandomState instance, default=None\n        The seed of the pseudo random number generator to use when shuffling\n        the data, i.e. getting the random vectors to initialize the algorithm.\n        Pass an int for reproducible results across multiple function calls.\n        See :term:`Glossary <random_state>`.\n\n    Notes\n    -----\n    This algorithm finds a (usually very good) approximate truncated\n    eigendecomposition using randomized methods to speed up the computations.\n\n    This method is particularly fast on large matrices on which\n    you wish to extract only a small number of components. In order to\n    obtain further speed up, `n_iter` can be set <=2 (at the cost of\n    loss of precision). To increase the precision it is recommended to\n    increase `n_oversamples`, up to `2*k-n_components` where k is the\n    effective rank. Usually, `n_components` is chosen to be greater than k\n    so increasing `n_oversamples` up to `n_components` should be enough.\n\n    Strategy \'value\': not implemented yet.\n    Algorithms 5.3, 5.4 and 5.5 in the Halko et al paper should provide good\n    candidates for a future implementation.\n\n    Strategy \'module\':\n    The principle is that for diagonalizable matrices, the singular values and\n    eigenvalues are related: if t is an eigenvalue of A, then :math:`|t|` is a\n    singular value of A. This method relies on a randomized SVD to find the n\n    singular components corresponding to the n singular values with largest\n    modules, and then uses the signs of the singular vectors to find the true\n    sign of t: if the sign of left and right singular vectors are different\n    then the corresponding eigenvalue is negative.\n\n    Returns\n    -------\n    eigvals : 1D array of shape (n_components,) containing the `n_components`\n        eigenvalues selected (see ``selection`` parameter).\n    eigvecs : 2D array of shape (M.shape[0], n_components) containing the\n        `n_components` eigenvectors corresponding to the `eigvals`, in the\n        corresponding order. Note that this follows the `scipy.linalg.eigh`\n        convention.\n\n    See Also\n    --------\n    :func:`randomized_svd`\n\n    References\n    ----------\n    * :arxiv:`"Finding structure with randomness:\n      Stochastic algorithms for constructing approximate matrix decompositions"\n      (Algorithm 4.3 for strategy \'module\') <0909.4061>`\n      Halko, et al. (2009)\n    '
    if selection == 'value':
        raise NotImplementedError()
    elif selection == 'module':
        (U, S, Vt) = randomized_svd(M, n_components=n_components, n_oversamples=n_oversamples, n_iter=n_iter, power_iteration_normalizer=power_iteration_normalizer, flip_sign=False, random_state=random_state)
        eigvecs = U[:, :n_components]
        eigvals = S[:n_components]
        diag_VtU = np.einsum('ji,ij->j', Vt[:n_components, :], U[:, :n_components])
        signs = np.sign(diag_VtU)
        eigvals = eigvals * signs
    else:
        raise ValueError('Invalid `selection`: %r' % selection)
    return (eigvals, eigvecs)

def weighted_mode(a, w, *, axis=0):
    if False:
        print('Hello World!')
    "Return an array of the weighted modal (most common) value in the passed array.\n\n    If there is more than one such value, only the first is returned.\n    The bin-count for the modal bins is also returned.\n\n    This is an extension of the algorithm in scipy.stats.mode.\n\n    Parameters\n    ----------\n    a : array-like of shape (n_samples,)\n        Array of which values to find mode(s).\n    w : array-like of shape (n_samples,)\n        Array of weights for each value.\n    axis : int, default=0\n        Axis along which to operate. Default is 0, i.e. the first axis.\n\n    Returns\n    -------\n    vals : ndarray\n        Array of modal values.\n    score : ndarray\n        Array of weighted counts for each mode.\n\n    See Also\n    --------\n    scipy.stats.mode: Calculates the Modal (most common) value of array elements\n        along specified axis.\n\n    Examples\n    --------\n    >>> from sklearn.utils.extmath import weighted_mode\n    >>> x = [4, 1, 4, 2, 4, 2]\n    >>> weights = [1, 1, 1, 1, 1, 1]\n    >>> weighted_mode(x, weights)\n    (array([4.]), array([3.]))\n\n    The value 4 appears three times: with uniform weights, the result is\n    simply the mode of the distribution.\n\n    >>> weights = [1, 3, 0.5, 1.5, 1, 2]  # deweight the 4's\n    >>> weighted_mode(x, weights)\n    (array([2.]), array([3.5]))\n\n    The value 2 has the highest score: it appears twice with weights of\n    1.5 and 2: the sum of these is 3.5.\n    "
    if axis is None:
        a = np.ravel(a)
        w = np.ravel(w)
        axis = 0
    else:
        a = np.asarray(a)
        w = np.asarray(w)
    if a.shape != w.shape:
        w = np.full(a.shape, w, dtype=w.dtype)
    scores = np.unique(np.ravel(a))
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)
    for score in scores:
        template = np.zeros(a.shape)
        ind = a == score
        template[ind] = w[ind]
        counts = np.expand_dims(np.sum(template, axis), axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent
    return (mostfrequent, oldcounts)

def cartesian(arrays, out=None):
    if False:
        while True:
            i = 10
    'Generate a cartesian product of input arrays.\n\n    Parameters\n    ----------\n    arrays : list of array-like\n        1-D arrays to form the cartesian product of.\n    out : ndarray of shape (M, len(arrays)), default=None\n        Array to place the cartesian product in.\n\n    Returns\n    -------\n    out : ndarray of shape (M, len(arrays))\n        Array containing the cartesian products formed of input arrays.\n        If not provided, the `dtype` of the output array is set to the most\n        permissive `dtype` of the input arrays, according to NumPy type\n        promotion.\n\n        .. versionadded:: 1.2\n           Add support for arrays of different types.\n\n    Notes\n    -----\n    This function may not be used on more than 32 arrays\n    because the underlying numpy functions do not support it.\n\n    Examples\n    --------\n    >>> from sklearn.utils.extmath import cartesian\n    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))\n    array([[1, 4, 6],\n           [1, 4, 7],\n           [1, 5, 6],\n           [1, 5, 7],\n           [2, 4, 6],\n           [2, 4, 7],\n           [2, 5, 6],\n           [2, 5, 7],\n           [3, 4, 6],\n           [3, 4, 7],\n           [3, 5, 6],\n           [3, 5, 7]])\n    '
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T
    if out is None:
        dtype = np.result_type(*arrays)
        out = np.empty_like(ix, dtype=dtype)
    for (n, arr) in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]
    return out

def svd_flip(u, v, u_based_decision=True):
    if False:
        print('Hello World!')
    "Sign correction to ensure deterministic output from SVD.\n\n    Adjusts the columns of u and the rows of v such that the loadings in the\n    columns in u that are largest in absolute value are always positive.\n\n    If u_based_decision is False, then the same sign correction is applied to\n    so that the rows in v that are largest in absolute value are always\n    positive.\n\n    Parameters\n    ----------\n    u : ndarray\n        Parameters u and v are the output of `linalg.svd` or\n        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner\n        dimensions so one can compute `np.dot(u * s, v)`.\n\n    v : ndarray\n        Parameters u and v are the output of `linalg.svd` or\n        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner\n        dimensions so one can compute `np.dot(u * s, v)`. The input v should\n        really be called vt to be consistent with scipy's output.\n\n    u_based_decision : bool, default=True\n        If True, use the columns of u as the basis for sign flipping.\n        Otherwise, use the rows of v. The choice of which variable to base the\n        decision on is generally algorithm dependent.\n\n    Returns\n    -------\n    u_adjusted : ndarray\n        Array u with adjusted columns and the same dimensions as u.\n\n    v_adjusted : ndarray\n        Array v with adjusted rows and the same dimensions as v.\n    "
    (xp, _) = get_namespace(u, v)
    device = getattr(u, 'device', None)
    if u_based_decision:
        max_abs_u_cols = xp.argmax(xp.abs(u.T), axis=1)
        shift = xp.arange(u.T.shape[0], device=device)
        indices = max_abs_u_cols + shift * u.T.shape[1]
        signs = xp.sign(xp.take(xp.reshape(u.T, (-1,)), indices, axis=0))
        u *= signs[np.newaxis, :]
        v *= signs[:, np.newaxis]
    else:
        max_abs_v_rows = xp.argmax(xp.abs(v), axis=1)
        shift = xp.arange(v.shape[0], device=device)
        indices = max_abs_v_rows + shift * v.shape[1]
        signs = xp.sign(xp.take(xp.reshape(v, (-1,)), indices))
        u *= signs[np.newaxis, :]
        v *= signs[:, np.newaxis]
    return (u, v)

@deprecated('The function `log_logistic` is deprecated and will be removed in 1.6. Use `-np.logaddexp(0, -x)` instead.')
def log_logistic(X, out=None):
    if False:
        while True:
            i = 10
    'Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.\n\n    This implementation is numerically stable and uses `-np.logaddexp(0, -x)`.\n\n    For the ordinary logistic function, use ``scipy.special.expit``.\n\n    Parameters\n    ----------\n    X : array-like of shape (M, N) or (M,)\n        Argument to the logistic function.\n\n    out : array-like of shape (M, N) or (M,), default=None\n        Preallocated output array.\n\n    Returns\n    -------\n    out : ndarray of shape (M, N) or (M,)\n        Log of the logistic function evaluated at every point in x.\n\n    Notes\n    -----\n    See the blog post describing this implementation:\n    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/\n    '
    X = check_array(X, dtype=np.float64, ensure_2d=False)
    if out is None:
        out = np.empty_like(X)
    np.logaddexp(0, -X, out=out)
    out *= -1
    return out

def softmax(X, copy=True):
    if False:
        while True:
            i = 10
    '\n    Calculate the softmax function.\n\n    The softmax function is calculated by\n    np.exp(X) / np.sum(np.exp(X), axis=1)\n\n    This will cause overflow when large values are exponentiated.\n    Hence the largest value in each row is subtracted from each data\n    point to prevent this.\n\n    Parameters\n    ----------\n    X : array-like of float of shape (M, N)\n        Argument to the logistic function.\n\n    copy : bool, default=True\n        Copy X or not.\n\n    Returns\n    -------\n    out : ndarray of shape (M, N)\n        Softmax function evaluated at every point in x.\n    '
    (xp, is_array_api_compliant) = get_namespace(X)
    if copy:
        X = xp.asarray(X, copy=True)
    max_prob = xp.reshape(xp.max(X, axis=1), (-1, 1))
    X -= max_prob
    if _is_numpy_namespace(xp):
        np.exp(X, out=np.asarray(X))
    else:
        X = xp.exp(X)
    sum_prob = xp.reshape(xp.sum(X, axis=1), (-1, 1))
    X /= sum_prob
    return X

def make_nonnegative(X, min_value=0):
    if False:
        i = 10
        return i + 15
    'Ensure `X.min()` >= `min_value`.\n\n    Parameters\n    ----------\n    X : array-like\n        The matrix to make non-negative.\n    min_value : float, default=0\n        The threshold value.\n\n    Returns\n    -------\n    array-like\n        The thresholded array.\n\n    Raises\n    ------\n    ValueError\n        When X is sparse.\n    '
    min_ = X.min()
    if min_ < min_value:
        if sparse.issparse(X):
            raise ValueError('Cannot make the data matrix nonnegative because it is sparse. Adding a value to every entry would make it no longer sparse.')
        X = X + (min_value - min_)
    return X

def _safe_accumulator_op(op, x, *args, **kwargs):
    if False:
        print('Hello World!')
    '\n    This function provides numpy accumulator functions with a float64 dtype\n    when used on a floating point input. This prevents accumulator overflow on\n    smaller floating point dtypes.\n\n    Parameters\n    ----------\n    op : function\n        A numpy accumulator function such as np.mean or np.sum.\n    x : ndarray\n        A numpy array to apply the accumulator function.\n    *args : positional arguments\n        Positional arguments passed to the accumulator function after the\n        input x.\n    **kwargs : keyword arguments\n        Keyword arguments passed to the accumulator function.\n\n    Returns\n    -------\n    result\n        The output of the accumulator function passed to this function.\n    '
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result

def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count, sample_weight=None):
    if False:
        for i in range(10):
            print('nop')
    'Calculate mean update and a Youngs and Cramer variance update.\n\n    If sample_weight is given, the weighted mean and variance is computed.\n\n    Update a given mean and (possibly) variance according to new data given\n    in X. last_mean is always required to compute the new mean.\n    If last_variance is None, no variance is computed and None return for\n    updated_variance.\n\n    From the paper "Algorithms for computing the sample variance: analysis and\n    recommendations", by Chan, Golub, and LeVeque.\n\n    Parameters\n    ----------\n    X : array-like of shape (n_samples, n_features)\n        Data to use for variance update.\n\n    last_mean : array-like of shape (n_features,)\n\n    last_variance : array-like of shape (n_features,)\n\n    last_sample_count : array-like of shape (n_features,)\n        The number of samples encountered until now if sample_weight is None.\n        If sample_weight is not None, this is the sum of sample_weight\n        encountered.\n\n    sample_weight : array-like of shape (n_samples,) or None\n        Sample weights. If None, compute the unweighted mean/variance.\n\n    Returns\n    -------\n    updated_mean : ndarray of shape (n_features,)\n\n    updated_variance : ndarray of shape (n_features,)\n        None if last_variance was None.\n\n    updated_sample_count : ndarray of shape (n_features,)\n\n    Notes\n    -----\n    NaNs are ignored during the algorithm.\n\n    References\n    ----------\n    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample\n        variance: recommendations, The American Statistician, Vol. 37, No. 3,\n        pp. 242-247\n\n    Also, see the sparse implementation of this in\n    `utils.sparsefuncs.incr_mean_variance_axis` and\n    `utils.sparsefuncs_fast.incr_mean_variance_axis0`\n    '
    last_sum = last_mean * last_sample_count
    X_nan_mask = np.isnan(X)
    if np.any(X_nan_mask):
        sum_op = np.nansum
    else:
        sum_op = np.sum
    if sample_weight is not None:
        new_sum = _safe_accumulator_op(np.matmul, sample_weight, np.where(X_nan_mask, 0, X))
        new_sample_count = _safe_accumulator_op(np.sum, sample_weight[:, None] * ~X_nan_mask, axis=0)
    else:
        new_sum = _safe_accumulator_op(sum_op, X, axis=0)
        n_samples = X.shape[0]
        new_sample_count = n_samples - np.sum(X_nan_mask, axis=0)
    updated_sample_count = last_sample_count + new_sample_count
    updated_mean = (last_sum + new_sum) / updated_sample_count
    if last_variance is None:
        updated_variance = None
    else:
        T = new_sum / new_sample_count
        temp = X - T
        if sample_weight is not None:
            correction = _safe_accumulator_op(np.matmul, sample_weight, np.where(X_nan_mask, 0, temp))
            temp **= 2
            new_unnormalized_variance = _safe_accumulator_op(np.matmul, sample_weight, np.where(X_nan_mask, 0, temp))
        else:
            correction = _safe_accumulator_op(sum_op, temp, axis=0)
            temp **= 2
            new_unnormalized_variance = _safe_accumulator_op(sum_op, temp, axis=0)
        new_unnormalized_variance -= correction ** 2 / new_sample_count
        last_unnormalized_variance = last_variance * last_sample_count
        with np.errstate(divide='ignore', invalid='ignore'):
            last_over_new_count = last_sample_count / new_sample_count
            updated_unnormalized_variance = last_unnormalized_variance + new_unnormalized_variance + last_over_new_count / updated_sample_count * (last_sum / last_over_new_count - new_sum) ** 2
        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count
    return (updated_mean, updated_variance, updated_sample_count)

def _deterministic_vector_sign_flip(u):
    if False:
        while True:
            i = 10
    'Modify the sign of vectors for reproducibility.\n\n    Flips the sign of elements of all the vectors (rows of u) such that\n    the absolute maximum element of each vector is positive.\n\n    Parameters\n    ----------\n    u : ndarray\n        Array with vectors as its rows.\n\n    Returns\n    -------\n    u_flipped : ndarray with same shape as u\n        Array with the sign flipped vectors as its rows.\n    '
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    if False:
        return 10
    'Use high precision for cumsum and check that final value matches sum.\n\n    Warns if the final cumulative sum does not match the sum (up to the chosen\n    tolerance).\n\n    Parameters\n    ----------\n    arr : array-like\n        To be cumulatively summed as flat.\n    axis : int, default=None\n        Axis along which the cumulative sum is computed.\n        The default (None) is to compute the cumsum over the flattened array.\n    rtol : float, default=1e-05\n        Relative tolerance, see ``np.allclose``.\n    atol : float, default=1e-08\n        Absolute tolerance, see ``np.allclose``.\n\n    Returns\n    -------\n    out : ndarray\n        Array with the cumulative sums along the chosen axis.\n    '
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.allclose(out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True):
        warnings.warn('cumsum was found to be unstable: its last element does not correspond to sum', RuntimeWarning)
    return out

def _nanaverage(a, weights=None):
    if False:
        return 10
    'Compute the weighted average, ignoring NaNs.\n\n    Parameters\n    ----------\n    a : ndarray\n        Array containing data to be averaged.\n    weights : array-like, default=None\n        An array of weights associated with the values in a. Each value in a\n        contributes to the average according to its associated weight. The\n        weights array can either be 1-D of the same shape as a. If `weights=None`,\n        then all data in a are assumed to have a weight equal to one.\n\n    Returns\n    -------\n    weighted_average : float\n        The weighted average.\n\n    Notes\n    -----\n    This wrapper to combine :func:`numpy.average` and :func:`numpy.nanmean`, so\n    that :func:`np.nan` values are ignored from the average and weights can\n    be passed. Note that when possible, we delegate to the prime methods.\n    '
    if len(a) == 0:
        return np.nan
    mask = np.isnan(a)
    if mask.all():
        return np.nan
    if weights is None:
        return np.nanmean(a)
    weights = np.array(weights, copy=False)
    (a, weights) = (a[~mask], weights[~mask])
    try:
        return np.average(a, weights=weights)
    except ZeroDivisionError:
        return np.average(a)