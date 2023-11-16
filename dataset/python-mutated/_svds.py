import os
import numpy as np
from .arpack import _arpack
from . import eigsh
from scipy._lib._util import check_random_state
from scipy.sparse.linalg._interface import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg
if os.environ.get('SCIPY_USE_PROPACK'):
    from scipy.sparse.linalg._svdp import _svdp
    HAS_PROPACK = True
else:
    HAS_PROPACK = False
from scipy.linalg import svd
arpack_int = _arpack.timing.nbx.dtype
__all__ = ['svds']

def _herm(x):
    if False:
        for i in range(10):
            print('nop')
    return x.T.conj()

def _iv(A, k, ncv, tol, which, v0, maxiter, return_singular, solver, random_state):
    if False:
        i = 10
        return i + 15
    solver = str(solver).lower()
    solvers = {'arpack', 'lobpcg', 'propack'}
    if solver not in solvers:
        raise ValueError(f'solver must be one of {solvers}.')
    A = aslinearoperator(A)
    if not (np.issubdtype(A.dtype, np.complexfloating) or np.issubdtype(A.dtype, np.floating)):
        message = '`A` must be of floating or complex floating data type.'
        raise ValueError(message)
    if np.prod(A.shape) == 0:
        message = '`A` must not be empty.'
        raise ValueError(message)
    kmax = min(A.shape) if solver == 'propack' else min(A.shape) - 1
    if int(k) != k or not 0 < k <= kmax:
        message = '`k` must be an integer satisfying `0 < k < min(A.shape)`.'
        raise ValueError(message)
    k = int(k)
    if solver == 'arpack' and ncv is not None:
        if int(ncv) != ncv or not k < ncv < min(A.shape):
            message = '`ncv` must be an integer satisfying `k < ncv < min(A.shape)`.'
            raise ValueError(message)
        ncv = int(ncv)
    if tol < 0 or not np.isfinite(tol):
        message = '`tol` must be a non-negative floating point value.'
        raise ValueError(message)
    tol = float(tol)
    which = str(which).upper()
    whichs = {'LM', 'SM'}
    if which not in whichs:
        raise ValueError(f'`which` must be in {whichs}.')
    if v0 is not None:
        v0 = np.atleast_1d(v0)
        if not (np.issubdtype(v0.dtype, np.complexfloating) or np.issubdtype(v0.dtype, np.floating)):
            message = '`v0` must be of floating or complex floating data type.'
            raise ValueError(message)
        shape = (A.shape[0],) if solver == 'propack' else (min(A.shape),)
        if v0.shape != shape:
            message = f'`v0` must have shape {shape}.'
            raise ValueError(message)
    if maxiter is not None and (int(maxiter) != maxiter or maxiter <= 0):
        message = '`maxiter` must be a positive integer.'
        raise ValueError(message)
    maxiter = int(maxiter) if maxiter is not None else maxiter
    rs_options = {True, False, 'vh', 'u'}
    if return_singular not in rs_options:
        raise ValueError(f'`return_singular_vectors` must be in {rs_options}.')
    random_state = check_random_state(random_state)
    return (A, k, ncv, tol, which, v0, maxiter, return_singular, solver, random_state)

def svds(A, k=6, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True, solver='arpack', random_state=None, options=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Partial singular value decomposition of a sparse matrix.\n\n    Compute the largest or smallest `k` singular values and corresponding\n    singular vectors of a sparse matrix `A`. The order in which the singular\n    values are returned is not guaranteed.\n\n    In the descriptions below, let ``M, N = A.shape``.\n\n    Parameters\n    ----------\n    A : ndarray, sparse matrix, or LinearOperator\n        Matrix to decompose of a floating point numeric dtype.\n    k : int, default: 6\n        Number of singular values and singular vectors to compute.\n        Must satisfy ``1 <= k <= kmax``, where ``kmax=min(M, N)`` for\n        ``solver=\'propack\'`` and ``kmax=min(M, N) - 1`` otherwise.\n    ncv : int, optional\n        When ``solver=\'arpack\'``, this is the number of Lanczos vectors\n        generated. See :ref:`\'arpack\' <sparse.linalg.svds-arpack>` for details.\n        When ``solver=\'lobpcg\'`` or ``solver=\'propack\'``, this parameter is\n        ignored.\n    tol : float, optional\n        Tolerance for singular values. Zero (default) means machine precision.\n    which : {\'LM\', \'SM\'}\n        Which `k` singular values to find: either the largest magnitude (\'LM\')\n        or smallest magnitude (\'SM\') singular values.\n    v0 : ndarray, optional\n        The starting vector for iteration; see method-specific\n        documentation (:ref:`\'arpack\' <sparse.linalg.svds-arpack>`,\n        :ref:`\'lobpcg\' <sparse.linalg.svds-lobpcg>`), or\n        :ref:`\'propack\' <sparse.linalg.svds-propack>` for details.\n    maxiter : int, optional\n        Maximum number of iterations; see method-specific\n        documentation (:ref:`\'arpack\' <sparse.linalg.svds-arpack>`,\n        :ref:`\'lobpcg\' <sparse.linalg.svds-lobpcg>`), or\n        :ref:`\'propack\' <sparse.linalg.svds-propack>` for details.\n    return_singular_vectors : {True, False, "u", "vh"}\n        Singular values are always computed and returned; this parameter\n        controls the computation and return of singular vectors.\n\n        - ``True``: return singular vectors.\n        - ``False``: do not return singular vectors.\n        - ``"u"``: if ``M <= N``, compute only the left singular vectors and\n          return ``None`` for the right singular vectors. Otherwise, compute\n          all singular vectors.\n        - ``"vh"``: if ``M > N``, compute only the right singular vectors and\n          return ``None`` for the left singular vectors. Otherwise, compute\n          all singular vectors.\n\n        If ``solver=\'propack\'``, the option is respected regardless of the\n        matrix shape.\n\n    solver :  {\'arpack\', \'propack\', \'lobpcg\'}, optional\n            The solver used.\n            :ref:`\'arpack\' <sparse.linalg.svds-arpack>`,\n            :ref:`\'lobpcg\' <sparse.linalg.svds-lobpcg>`, and\n            :ref:`\'propack\' <sparse.linalg.svds-propack>` are supported.\n            Default: `\'arpack\'`.\n    random_state : {None, int, `numpy.random.Generator`,\n                    `numpy.random.RandomState`}, optional\n\n        Pseudorandom number generator state used to generate resamples.\n\n        If `random_state` is ``None`` (or `np.random`), the\n        `numpy.random.RandomState` singleton is used.\n        If `random_state` is an int, a new ``RandomState`` instance is used,\n        seeded with `random_state`.\n        If `random_state` is already a ``Generator`` or ``RandomState``\n        instance then that instance is used.\n    options : dict, optional\n        A dictionary of solver-specific options. No solver-specific options\n        are currently supported; this parameter is reserved for future use.\n\n    Returns\n    -------\n    u : ndarray, shape=(M, k)\n        Unitary matrix having left singular vectors as columns.\n    s : ndarray, shape=(k,)\n        The singular values.\n    vh : ndarray, shape=(k, N)\n        Unitary matrix having right singular vectors as rows.\n\n    Notes\n    -----\n    This is a naive implementation using ARPACK or LOBPCG as an eigensolver\n    on the matrix ``A.conj().T @ A`` or ``A @ A.conj().T``, depending on\n    which one is smaller size, followed by the Rayleigh-Ritz method\n    as postprocessing; see\n    Using the normal matrix, in Rayleigh-Ritz method, (2022, Nov. 19),\n    Wikipedia, https://w.wiki/4zms.\n\n    Alternatively, the PROPACK solver can be called.\n\n    Choices of the input matrix `A` numeric dtype may be limited.\n    Only ``solver="lobpcg"`` supports all floating point dtypes\n    real: \'np.float32\', \'np.float64\', \'np.longdouble\' and\n    complex: \'np.complex64\', \'np.complex128\', \'np.clongdouble\'.\n    The ``solver="arpack"`` supports only\n    \'np.float32\', \'np.float64\', and \'np.complex128\'.\n\n    Examples\n    --------\n    Construct a matrix `A` from singular values and vectors.\n\n    >>> import numpy as np\n    >>> from scipy import sparse, linalg, stats\n    >>> from scipy.sparse.linalg import svds, aslinearoperator, LinearOperator\n\n    Construct a dense matrix `A` from singular values and vectors.\n\n    >>> rng = np.random.default_rng(258265244568965474821194062361901728911)\n    >>> orthogonal = stats.ortho_group.rvs(10, random_state=rng)\n    >>> s = [1e-3, 1, 2, 3, 4]  # non-zero singular values\n    >>> u = orthogonal[:, :5]         # left singular vectors\n    >>> vT = orthogonal[:, 5:].T      # right singular vectors\n    >>> A = u @ np.diag(s) @ vT\n\n    With only four singular values/vectors, the SVD approximates the original\n    matrix.\n\n    >>> u4, s4, vT4 = svds(A, k=4)\n    >>> A4 = u4 @ np.diag(s4) @ vT4\n    >>> np.allclose(A4, A, atol=1e-3)\n    True\n\n    With all five non-zero singular values/vectors, we can reproduce\n    the original matrix more accurately.\n\n    >>> u5, s5, vT5 = svds(A, k=5)\n    >>> A5 = u5 @ np.diag(s5) @ vT5\n    >>> np.allclose(A5, A)\n    True\n\n    The singular values match the expected singular values.\n\n    >>> np.allclose(s5, s)\n    True\n\n    Since the singular values are not close to each other in this example,\n    every singular vector matches as expected up to a difference in sign.\n\n    >>> (np.allclose(np.abs(u5), np.abs(u)) and\n    ...  np.allclose(np.abs(vT5), np.abs(vT)))\n    True\n\n    The singular vectors are also orthogonal.\n\n    >>> (np.allclose(u5.T @ u5, np.eye(5)) and\n    ...  np.allclose(vT5 @ vT5.T, np.eye(5)))\n    True\n\n    If there are (nearly) multiple singular values, the corresponding\n    individual singular vectors may be unstable, but the whole invariant\n    subspace containing all such singular vectors is computed accurately\n    as can be measured by angles between subspaces via \'subspace_angles\'.\n\n    >>> rng = np.random.default_rng(178686584221410808734965903901790843963)\n    >>> s = [1, 1 + 1e-6]  # non-zero singular values\n    >>> u, _ = np.linalg.qr(rng.standard_normal((99, 2)))\n    >>> v, _ = np.linalg.qr(rng.standard_normal((99, 2)))\n    >>> vT = v.T\n    >>> A = u @ np.diag(s) @ vT\n    >>> A = A.astype(np.float32)\n    >>> u2, s2, vT2 = svds(A, k=2, random_state=rng)\n    >>> np.allclose(s2, s)\n    True\n\n    The angles between the individual exact and computed singular vectors\n    may not be so small. To check use:\n\n    >>> (linalg.subspace_angles(u2[:, :1], u[:, :1]) +\n    ...  linalg.subspace_angles(u2[:, 1:], u[:, 1:]))\n    array([0.06562513])  # may vary\n    >>> (linalg.subspace_angles(vT2[:1, :].T, vT[:1, :].T) +\n    ...  linalg.subspace_angles(vT2[1:, :].T, vT[1:, :].T))\n    array([0.06562507])  # may vary\n\n    As opposed to the angles between the 2-dimensional invariant subspaces\n    that these vectors span, which are small for rights singular vectors\n\n    >>> linalg.subspace_angles(u2, u).sum() < 1e-6\n    True\n\n    as well as for left singular vectors.\n\n    >>> linalg.subspace_angles(vT2.T, vT.T).sum() < 1e-6\n    True\n\n    The next example follows that of \'sklearn.decomposition.TruncatedSVD\'.\n\n    >>> rng = np.random.RandomState(0)\n    >>> X_dense = rng.random(size=(100, 100))\n    >>> X_dense[:, 2 * np.arange(50)] = 0\n    >>> X = sparse.csr_matrix(X_dense)\n    >>> _, singular_values, _ = svds(X, k=5, random_state=rng)\n    >>> print(singular_values)\n    [ 4.3293...  4.4491...  4.5420...  4.5987... 35.2410...]\n\n    The function can be called without the transpose of the input matrix\n    ever explicitly constructed.\n\n    >>> rng = np.random.default_rng(102524723947864966825913730119128190974)\n    >>> G = sparse.rand(8, 9, density=0.5, random_state=rng)\n    >>> Glo = aslinearoperator(G)\n    >>> _, singular_values_svds, _ = svds(Glo, k=5, random_state=rng)\n    >>> _, singular_values_svd, _ = linalg.svd(G.toarray())\n    >>> np.allclose(singular_values_svds, singular_values_svd[-4::-1])\n    True\n\n    The most memory efficient scenario is where neither\n    the original matrix, nor its transpose, is explicitly constructed.\n    Our example computes the smallest singular values and vectors\n    of \'LinearOperator\' constructed from the numpy function \'np.diff\' used\n    column-wise to be consistent with \'LinearOperator\' operating on columns.\n\n    >>> diff0 = lambda a: np.diff(a, axis=0)\n\n    Let us create the matrix from \'diff0\' to be used for validation only.\n\n    >>> n = 5  # The dimension of the space.\n    >>> M_from_diff0 = diff0(np.eye(n))\n    >>> print(M_from_diff0.astype(int))\n    [[-1  1  0  0  0]\n     [ 0 -1  1  0  0]\n     [ 0  0 -1  1  0]\n     [ 0  0  0 -1  1]]\n\n    The matrix \'M_from_diff0\' is bi-diagonal and could be alternatively\n    created directly by\n\n    >>> M = - np.eye(n - 1, n, dtype=int)\n    >>> np.fill_diagonal(M[:,1:], 1)\n    >>> np.allclose(M, M_from_diff0)\n    True\n\n    Its transpose\n\n    >>> print(M.T)\n    [[-1  0  0  0]\n     [ 1 -1  0  0]\n     [ 0  1 -1  0]\n     [ 0  0  1 -1]\n     [ 0  0  0  1]]\n\n    can be viewed as the incidence matrix; see\n    Incidence matrix, (2022, Nov. 19), Wikipedia, https://w.wiki/5YXU,\n    of a linear graph with 5 vertices and 4 edges. The 5x5 normal matrix\n    ``M.T @ M`` thus is\n\n    >>> print(M.T @ M)\n    [[ 1 -1  0  0  0]\n     [-1  2 -1  0  0]\n     [ 0 -1  2 -1  0]\n     [ 0  0 -1  2 -1]\n     [ 0  0  0 -1  1]]\n\n    the graph Laplacian, while the actually used in \'svds\' smaller size\n    4x4 normal matrix ``M @ M.T``\n\n    >>> print(M @ M.T)\n    [[ 2 -1  0  0]\n     [-1  2 -1  0]\n     [ 0 -1  2 -1]\n     [ 0  0 -1  2]]\n\n    is the so-called edge-based Laplacian; see\n    Symmetric Laplacian via the incidence matrix, in Laplacian matrix,\n    (2022, Nov. 19), Wikipedia, https://w.wiki/5YXW.\n\n    The \'LinearOperator\' setup needs the options \'rmatvec\' and \'rmatmat\'\n    of multiplication by the matrix transpose ``M.T``, but we want to be\n    matrix-free to save memory, so knowing how ``M.T`` looks like, we\n    manually construct the following function to be\n    used in ``rmatmat=diff0t``.\n\n    >>> def diff0t(a):\n    ...     if a.ndim == 1:\n    ...         a = a[:,np.newaxis]  # Turn 1D into 2D array\n    ...     d = np.zeros((a.shape[0] + 1, a.shape[1]), dtype=a.dtype)\n    ...     d[0, :] = - a[0, :]\n    ...     d[1:-1, :] = a[0:-1, :] - a[1:, :]\n    ...     d[-1, :] = a[-1, :]\n    ...     return d\n\n    We check that our function \'diff0t\' for the matrix transpose is valid.\n\n    >>> np.allclose(M.T, diff0t(np.eye(n-1)))\n    True\n\n    Now we setup our matrix-free \'LinearOperator\' called \'diff0_func_aslo\'\n    and for validation the matrix-based \'diff0_matrix_aslo\'.\n\n    >>> def diff0_func_aslo_def(n):\n    ...     return LinearOperator(matvec=diff0,\n    ...                           matmat=diff0,\n    ...                           rmatvec=diff0t,\n    ...                           rmatmat=diff0t,\n    ...                           shape=(n - 1, n))\n    >>> diff0_func_aslo = diff0_func_aslo_def(n)\n    >>> diff0_matrix_aslo = aslinearoperator(M_from_diff0)\n\n    And validate both the matrix and its transpose in \'LinearOperator\'.\n\n    >>> np.allclose(diff0_func_aslo(np.eye(n)),\n    ...             diff0_matrix_aslo(np.eye(n)))\n    True\n    >>> np.allclose(diff0_func_aslo.T(np.eye(n-1)),\n    ...             diff0_matrix_aslo.T(np.eye(n-1)))\n    True\n\n    Having the \'LinearOperator\' setup validated, we run the solver.\n\n    >>> n = 100\n    >>> diff0_func_aslo = diff0_func_aslo_def(n)\n    >>> u, s, vT = svds(diff0_func_aslo, k=3, which=\'SM\')\n\n    The singular values squared and the singular vectors are known\n    explicitly; see\n    Pure Dirichlet boundary conditions, in\n    Eigenvalues and eigenvectors of the second derivative,\n    (2022, Nov. 19), Wikipedia, https://w.wiki/5YX6,\n    since \'diff\' corresponds to first\n    derivative, and its smaller size n-1 x n-1 normal matrix\n    ``M @ M.T`` represent the discrete second derivative with the Dirichlet\n    boundary conditions. We use these analytic expressions for validation.\n\n    >>> se = 2. * np.sin(np.pi * np.arange(1, 4) / (2. * n))\n    >>> ue = np.sqrt(2 / n) * np.sin(np.pi * np.outer(np.arange(1, n),\n    ...                              np.arange(1, 4)) / n)\n    >>> np.allclose(s, se, atol=1e-3)\n    True\n    >>> print(np.allclose(np.abs(u), np.abs(ue), atol=1e-6))\n    True\n\n    '
    args = _iv(A, k, ncv, tol, which, v0, maxiter, return_singular_vectors, solver, random_state)
    (A, k, ncv, tol, which, v0, maxiter, return_singular_vectors, solver, random_state) = args
    largest = which == 'LM'
    (n, m) = A.shape
    if n >= m:
        X_dot = A.matvec
        X_matmat = A.matmat
        XH_dot = A.rmatvec
        XH_mat = A.rmatmat
        transpose = False
    else:
        X_dot = A.rmatvec
        X_matmat = A.rmatmat
        XH_dot = A.matvec
        XH_mat = A.matmat
        transpose = True
        dtype = getattr(A, 'dtype', None)
        if dtype is None:
            dtype = A.dot(np.zeros([m, 1])).dtype

    def matvec_XH_X(x):
        if False:
            while True:
                i = 10
        return XH_dot(X_dot(x))

    def matmat_XH_X(x):
        if False:
            return 10
        return XH_mat(X_matmat(x))
    XH_X = LinearOperator(matvec=matvec_XH_X, dtype=A.dtype, matmat=matmat_XH_X, shape=(min(A.shape), min(A.shape)))
    if solver == 'lobpcg':
        if k == 1 and v0 is not None:
            X = np.reshape(v0, (-1, 1))
        else:
            X = random_state.standard_normal(size=(min(A.shape), k))
        (_, eigvec) = lobpcg(XH_X, X, tol=tol ** 2, maxiter=maxiter, largest=largest)
    elif solver == 'propack':
        if not HAS_PROPACK:
            raise ValueError("`solver='propack'` is opt-in due to potential issues on Windows, it can be enabled by setting the `SCIPY_USE_PROPACK` environment variable before importing scipy")
        jobu = return_singular_vectors in {True, 'u'}
        jobv = return_singular_vectors in {True, 'vh'}
        irl_mode = which == 'SM'
        res = _svdp(A, k=k, tol=tol ** 2, which=which, maxiter=None, compute_u=jobu, compute_v=jobv, irl_mode=irl_mode, kmax=maxiter, v0=v0, random_state=random_state)
        (u, s, vh, _) = res
        s = s[::-1]
        u = u[:, ::-1]
        vh = vh[::-1]
        u = u if jobu else None
        vh = vh if jobv else None
        if return_singular_vectors:
            return (u, s, vh)
        else:
            return s
    elif solver == 'arpack' or solver is None:
        if v0 is None:
            v0 = random_state.standard_normal(size=(min(A.shape),))
        (_, eigvec) = eigsh(XH_X, k=k, tol=tol ** 2, maxiter=maxiter, ncv=ncv, which=which, v0=v0)
        (eigvec, _) = np.linalg.qr(eigvec)
    Av = X_matmat(eigvec)
    if not return_singular_vectors:
        s = svd(Av, compute_uv=False, overwrite_a=True)
        return s[::-1]
    (u, s, vh) = svd(Av, full_matrices=False, overwrite_a=True)
    u = u[:, ::-1]
    s = s[::-1]
    vh = vh[::-1]
    jobu = return_singular_vectors in {True, 'u'}
    jobv = return_singular_vectors in {True, 'vh'}
    if transpose:
        u_tmp = eigvec @ _herm(vh) if jobu else None
        vh = _herm(u) if jobv else None
        u = u_tmp
    else:
        if not jobu:
            u = None
        vh = vh @ _herm(eigvec) if jobv else None
    return (u, s, vh)