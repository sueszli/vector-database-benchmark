"""Locally Linear Embedding"""
from numbers import Integral, Real
import numpy as np
from scipy.linalg import eigh, qr, solve, svd
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin, _fit_context, _UnstableArchMixin
from ..neighbors import NearestNeighbors
from ..utils import check_array, check_random_state
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import stable_cumsum
from ..utils.validation import FLOAT_DTYPES, check_is_fitted

def barycenter_weights(X, Y, indices, reg=0.001):
    if False:
        while True:
            i = 10
    'Compute barycenter weights of X from Y along the first axis\n\n    We estimate the weights to assign to each point in Y[indices] to recover\n    the point X[i]. The barycenter weights sum to 1.\n\n    Parameters\n    ----------\n    X : array-like, shape (n_samples, n_dim)\n\n    Y : array-like, shape (n_samples, n_dim)\n\n    indices : array-like, shape (n_samples, n_dim)\n            Indices of the points in Y used to compute the barycenter\n\n    reg : float, default=1e-3\n        Amount of regularization to add for the problem to be\n        well-posed in the case of n_neighbors > n_dim\n\n    Returns\n    -------\n    B : array-like, shape (n_samples, n_neighbors)\n\n    Notes\n    -----\n    See developers note for more information.\n    '
    X = check_array(X, dtype=FLOAT_DTYPES)
    Y = check_array(Y, dtype=FLOAT_DTYPES)
    indices = check_array(indices, dtype=int)
    (n_samples, n_neighbors) = indices.shape
    assert X.shape[0] == n_samples
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)
    for (i, ind) in enumerate(indices):
        A = Y[ind]
        C = A - X[i]
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::n_neighbors + 1] += R
        w = solve(G, v, assume_a='pos')
        B[i, :] = w / np.sum(w)
    return B

def barycenter_kneighbors_graph(X, n_neighbors, reg=0.001, n_jobs=None):
    if False:
        i = 10
        return i + 15
    "Computes the barycenter weighted graph of k-Neighbors for points in X\n\n    Parameters\n    ----------\n    X : {array-like, NearestNeighbors}\n        Sample data, shape = (n_samples, n_features), in the form of a\n        numpy array or a NearestNeighbors object.\n\n    n_neighbors : int\n        Number of neighbors for each sample.\n\n    reg : float, default=1e-3\n        Amount of regularization when solving the least-squares\n        problem. Only relevant if mode='barycenter'. If None, use the\n        default.\n\n    n_jobs : int or None, default=None\n        The number of parallel jobs to run for neighbors search.\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    Returns\n    -------\n    A : sparse matrix in CSR format, shape = [n_samples, n_samples]\n        A[i, j] is assigned the weight of edge that connects i to j.\n\n    See Also\n    --------\n    sklearn.neighbors.kneighbors_graph\n    sklearn.neighbors.radius_neighbors_graph\n    "
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs).fit(X)
    X = knn._fit_X
    n_samples = knn.n_samples_fit_
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X, ind, reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr), shape=(n_samples, n_samples))

def null_space(M, k, k_skip=1, eigen_solver='arpack', tol=1e-06, max_iter=100, random_state=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Find the null space of a matrix M.\n\n    Parameters\n    ----------\n    M : {array, matrix, sparse matrix, LinearOperator}\n        Input covariance matrix: should be symmetric positive semi-definite\n\n    k : int\n        Number of eigenvalues/vectors to return\n\n    k_skip : int, default=1\n        Number of low eigenvalues to skip.\n\n    eigen_solver : {'auto', 'arpack', 'dense'}, default='arpack'\n        auto : algorithm will attempt to choose the best method for input data\n        arpack : use arnoldi iteration in shift-invert mode.\n                    For this method, M may be a dense matrix, sparse matrix,\n                    or general linear operator.\n                    Warning: ARPACK can be unstable for some problems.  It is\n                    best to try several random seeds in order to check results.\n        dense  : use standard dense matrix operations for the eigenvalue\n                    decomposition.  For this method, M must be an array\n                    or matrix type.  This method should be avoided for\n                    large problems.\n\n    tol : float, default=1e-6\n        Tolerance for 'arpack' method.\n        Not used if eigen_solver=='dense'.\n\n    max_iter : int, default=100\n        Maximum number of iterations for 'arpack' method.\n        Not used if eigen_solver=='dense'\n\n    random_state : int, RandomState instance, default=None\n        Determines the random number generator when ``solver`` == 'arpack'.\n        Pass an int for reproducible results across multiple function calls.\n        See :term:`Glossary <random_state>`.\n    "
    if eigen_solver == 'auto':
        if M.shape[0] > 200 and k + k_skip < 10:
            eigen_solver = 'arpack'
        else:
            eigen_solver = 'dense'
    if eigen_solver == 'arpack':
        v0 = _init_arpack_v0(M.shape[0], random_state)
        try:
            (eigen_values, eigen_vectors) = eigsh(M, k + k_skip, sigma=0.0, tol=tol, maxiter=max_iter, v0=v0)
        except RuntimeError as e:
            raise ValueError("Error in determining null-space with ARPACK. Error message: '%s'. Note that eigen_solver='arpack' can fail when the weight matrix is singular or otherwise ill-behaved. In that case, eigen_solver='dense' is recommended. See online documentation for more information." % e) from e
        return (eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:]))
    elif eigen_solver == 'dense':
        if hasattr(M, 'toarray'):
            M = M.toarray()
        (eigen_values, eigen_vectors) = eigh(M, subset_by_index=(k_skip, k + k_skip - 1), overwrite_a=True)
        index = np.argsort(np.abs(eigen_values))
        return (eigen_vectors[:, index], np.sum(eigen_values))
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)

def locally_linear_embedding(X, *, n_neighbors, n_components, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12, random_state=None, n_jobs=None):
    if False:
        i = 10
        return i + 15
    "Perform a Locally Linear Embedding analysis on the data.\n\n    Read more in the :ref:`User Guide <locally_linear_embedding>`.\n\n    Parameters\n    ----------\n    X : {array-like, NearestNeighbors}\n        Sample data, shape = (n_samples, n_features), in the form of a\n        numpy array or a NearestNeighbors object.\n\n    n_neighbors : int\n        Number of neighbors to consider for each point.\n\n    n_components : int\n        Number of coordinates for the manifold.\n\n    reg : float, default=1e-3\n        Regularization constant, multiplies the trace of the local covariance\n        matrix of the distances.\n\n    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'\n        auto : algorithm will attempt to choose the best method for input data\n\n        arpack : use arnoldi iteration in shift-invert mode.\n                    For this method, M may be a dense matrix, sparse matrix,\n                    or general linear operator.\n                    Warning: ARPACK can be unstable for some problems.  It is\n                    best to try several random seeds in order to check results.\n\n        dense  : use standard dense matrix operations for the eigenvalue\n                    decomposition.  For this method, M must be an array\n                    or matrix type.  This method should be avoided for\n                    large problems.\n\n    tol : float, default=1e-6\n        Tolerance for 'arpack' method\n        Not used if eigen_solver=='dense'.\n\n    max_iter : int, default=100\n        Maximum number of iterations for the arpack solver.\n\n    method : {'standard', 'hessian', 'modified', 'ltsa'}, default='standard'\n        standard : use the standard locally linear embedding algorithm.\n                   see reference [1]_\n        hessian  : use the Hessian eigenmap method.  This method requires\n                   n_neighbors > n_components * (1 + (n_components + 1) / 2.\n                   see reference [2]_\n        modified : use the modified locally linear embedding algorithm.\n                   see reference [3]_\n        ltsa     : use local tangent space alignment algorithm\n                   see reference [4]_\n\n    hessian_tol : float, default=1e-4\n        Tolerance for Hessian eigenmapping method.\n        Only used if method == 'hessian'.\n\n    modified_tol : float, default=1e-12\n        Tolerance for modified LLE method.\n        Only used if method == 'modified'.\n\n    random_state : int, RandomState instance, default=None\n        Determines the random number generator when ``solver`` == 'arpack'.\n        Pass an int for reproducible results across multiple function calls.\n        See :term:`Glossary <random_state>`.\n\n    n_jobs : int or None, default=None\n        The number of parallel jobs to run for neighbors search.\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    Returns\n    -------\n    Y : array-like, shape [n_samples, n_components]\n        Embedding vectors.\n\n    squared_error : float\n        Reconstruction error for the embedding vectors. Equivalent to\n        ``norm(Y - W Y, 'fro')**2``, where W are the reconstruction weights.\n\n    References\n    ----------\n\n    .. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction\n        by locally linear embedding.  Science 290:2323 (2000).\n    .. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally\n        linear embedding techniques for high-dimensional data.\n        Proc Natl Acad Sci U S A.  100:5591 (2003).\n    .. [3] `Zhang, Z. & Wang, J. MLLE: Modified Locally Linear\n        Embedding Using Multiple Weights.\n        <https://citeseerx.ist.psu.edu/doc_view/pid/0b060fdbd92cbcc66b383bcaa9ba5e5e624d7ee3>`_\n    .. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear\n        dimensionality reduction via tangent space alignment.\n        Journal of Shanghai Univ.  8:406 (2004)\n    "
    if eigen_solver not in ('auto', 'arpack', 'dense'):
        raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)
    if method not in ('standard', 'hessian', 'modified', 'ltsa'):
        raise ValueError("unrecognized method '%s'" % method)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(X)
    X = nbrs._fit_X
    (N, d_in) = X.shape
    if n_components > d_in:
        raise ValueError('output dimension must be less than or equal to input dimension')
    if n_neighbors >= N:
        raise ValueError('Expected n_neighbors <= n_samples,  but n_samples = %d, n_neighbors = %d' % (N, n_neighbors))
    if n_neighbors <= 0:
        raise ValueError('n_neighbors must be positive')
    M_sparse = eigen_solver != 'dense'
    if method == 'standard':
        W = barycenter_kneighbors_graph(nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)
        if M_sparse:
            M = eye(*W.shape, format=W.format) - W
            M = (M.T * M).tocsr()
        else:
            M = (W.T * W - W.T - W).toarray()
            M.flat[::M.shape[0] + 1] += 1
    elif method == 'hessian':
        dp = n_components * (n_components + 1) // 2
        if n_neighbors <= n_components + dp:
            raise ValueError("for method='hessian', n_neighbors must be greater than [n_components * (n_components + 3) / 2]")
        neighbors = nbrs.kneighbors(X, n_neighbors=n_neighbors + 1, return_distance=False)
        neighbors = neighbors[:, 1:]
        Yi = np.empty((n_neighbors, 1 + n_components + dp), dtype=np.float64)
        Yi[:, 0] = 1
        M = np.zeros((N, N), dtype=np.float64)
        use_svd = n_neighbors > d_in
        for i in range(N):
            Gi = X[neighbors[i]]
            Gi -= Gi.mean(0)
            if use_svd:
                U = svd(Gi, full_matrices=0)[0]
            else:
                Ci = np.dot(Gi, Gi.T)
                U = eigh(Ci)[1][:, ::-1]
            Yi[:, 1:1 + n_components] = U[:, :n_components]
            j = 1 + n_components
            for k in range(n_components):
                Yi[:, j:j + n_components - k] = U[:, k:k + 1] * U[:, k:n_components]
                j += n_components - k
            (Q, R) = qr(Yi)
            w = Q[:, n_components + 1:]
            S = w.sum(0)
            S[np.where(abs(S) < hessian_tol)] = 1
            w /= S
            (nbrs_x, nbrs_y) = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] += np.dot(w, w.T)
        if M_sparse:
            M = csr_matrix(M)
    elif method == 'modified':
        if n_neighbors < n_components:
            raise ValueError('modified LLE requires n_neighbors >= n_components')
        neighbors = nbrs.kneighbors(X, n_neighbors=n_neighbors + 1, return_distance=False)
        neighbors = neighbors[:, 1:]
        V = np.zeros((N, n_neighbors, n_neighbors))
        nev = min(d_in, n_neighbors)
        evals = np.zeros([N, nev])
        use_svd = n_neighbors > d_in
        if use_svd:
            for i in range(N):
                X_nbrs = X[neighbors[i]] - X[i]
                (V[i], evals[i], _) = svd(X_nbrs, full_matrices=True)
            evals **= 2
        else:
            for i in range(N):
                X_nbrs = X[neighbors[i]] - X[i]
                C_nbrs = np.dot(X_nbrs, X_nbrs.T)
                (evi, vi) = eigh(C_nbrs)
                evals[i] = evi[::-1]
                V[i] = vi[:, ::-1]
        reg = 0.001 * evals.sum(1)
        tmp = np.dot(V.transpose(0, 2, 1), np.ones(n_neighbors))
        tmp[:, :nev] /= evals + reg[:, None]
        tmp[:, nev:] /= reg[:, None]
        w_reg = np.zeros((N, n_neighbors))
        for i in range(N):
            w_reg[i] = np.dot(V[i], tmp[i])
        w_reg /= w_reg.sum(1)[:, None]
        rho = evals[:, n_components:].sum(1) / evals[:, :n_components].sum(1)
        eta = np.median(rho)
        s_range = np.zeros(N, dtype=int)
        evals_cumsum = stable_cumsum(evals, 1)
        eta_range = evals_cumsum[:, -1:] / evals_cumsum[:, :-1] - 1
        for i in range(N):
            s_range[i] = np.searchsorted(eta_range[i, ::-1], eta)
        s_range += n_neighbors - nev
        M = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            s_i = s_range[i]
            Vi = V[i, :, n_neighbors - s_i:]
            alpha_i = np.linalg.norm(Vi.sum(0)) / np.sqrt(s_i)
            h = np.full(s_i, alpha_i) - np.dot(Vi.T, np.ones(n_neighbors))
            norm_h = np.linalg.norm(h)
            if norm_h < modified_tol:
                h *= 0
            else:
                h /= norm_h
            Wi = Vi - 2 * np.outer(np.dot(Vi, h), h) + (1 - alpha_i) * w_reg[i, :, None]
            (nbrs_x, nbrs_y) = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] += np.dot(Wi, Wi.T)
            Wi_sum1 = Wi.sum(1)
            M[i, neighbors[i]] -= Wi_sum1
            M[neighbors[i], i] -= Wi_sum1
            M[i, i] += s_i
        if M_sparse:
            M = csr_matrix(M)
    elif method == 'ltsa':
        neighbors = nbrs.kneighbors(X, n_neighbors=n_neighbors + 1, return_distance=False)
        neighbors = neighbors[:, 1:]
        M = np.zeros((N, N))
        use_svd = n_neighbors > d_in
        for i in range(N):
            Xi = X[neighbors[i]]
            Xi -= Xi.mean(0)
            if use_svd:
                v = svd(Xi, full_matrices=True)[0]
            else:
                Ci = np.dot(Xi, Xi.T)
                v = eigh(Ci)[1][:, ::-1]
            Gi = np.zeros((n_neighbors, n_components + 1))
            Gi[:, 1:] = v[:, :n_components]
            Gi[:, 0] = 1.0 / np.sqrt(n_neighbors)
            GiGiT = np.dot(Gi, Gi.T)
            (nbrs_x, nbrs_y) = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] -= GiGiT
            M[neighbors[i], neighbors[i]] += 1
    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver, tol=tol, max_iter=max_iter, random_state=random_state)

class LocallyLinearEmbedding(ClassNamePrefixFeaturesOutMixin, TransformerMixin, _UnstableArchMixin, BaseEstimator):
    """Locally Linear Embedding.

    Read more in the :ref:`User Guide <locally_linear_embedding>`.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to consider for each point.

    n_components : int, default=2
        Number of coordinates for the manifold.

    reg : float, default=1e-3
        Regularization constant, multiplies the trace of the local covariance
        matrix of the distances.

    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        The solver used to compute the eigenvectors. The available options are:

        - `'auto'` : algorithm will attempt to choose the best method for input
          data.
        - `'arpack'` : use arnoldi iteration in shift-invert mode. For this
          method, M may be a dense matrix, sparse matrix, or general linear
          operator.
        - `'dense'`  : use standard dense matrix operations for the eigenvalue
          decomposition. For this method, M must be an array or matrix type.
          This method should be avoided for large problems.

        .. warning::
           ARPACK can be unstable for some problems.  It is best to try several
           random seeds in order to check results.

    tol : float, default=1e-6
        Tolerance for 'arpack' method
        Not used if eigen_solver=='dense'.

    max_iter : int, default=100
        Maximum number of iterations for the arpack solver.
        Not used if eigen_solver=='dense'.

    method : {'standard', 'hessian', 'modified', 'ltsa'}, default='standard'
        - `standard`: use the standard locally linear embedding algorithm. see
          reference [1]_
        - `hessian`: use the Hessian eigenmap method. This method requires
          ``n_neighbors > n_components * (1 + (n_components + 1) / 2``. see
          reference [2]_
        - `modified`: use the modified locally linear embedding algorithm.
          see reference [3]_
        - `ltsa`: use local tangent space alignment algorithm. see
          reference [4]_

    hessian_tol : float, default=1e-4
        Tolerance for Hessian eigenmapping method.
        Only used if ``method == 'hessian'``.

    modified_tol : float, default=1e-12
        Tolerance for modified LLE method.
        Only used if ``method == 'modified'``.

    neighbors_algorithm : {'auto', 'brute', 'kd_tree', 'ball_tree'},                           default='auto'
        Algorithm to use for nearest neighbors search, passed to
        :class:`~sklearn.neighbors.NearestNeighbors` instance.

    random_state : int, RandomState instance, default=None
        Determines the random number generator when
        ``eigen_solver`` == 'arpack'. Pass an int for reproducible results
        across multiple function calls. See :term:`Glossary <random_state>`.

    n_jobs : int or None, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    embedding_ : array-like, shape [n_samples, n_components]
        Stores the embedding vectors

    reconstruction_error_ : float
        Reconstruction error associated with `embedding_`

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    nbrs_ : NearestNeighbors object
        Stores nearest neighbors instance, including BallTree or KDtree
        if applicable.

    See Also
    --------
    SpectralEmbedding : Spectral embedding for non-linear dimensionality
        reduction.
    TSNE : Distributed Stochastic Neighbor Embedding.

    References
    ----------

    .. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
        by locally linear embedding.  Science 290:2323 (2000).
    .. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
        linear embedding techniques for high-dimensional data.
        Proc Natl Acad Sci U S A.  100:5591 (2003).
    .. [3] `Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
        Embedding Using Multiple Weights.
        <https://citeseerx.ist.psu.edu/doc_view/pid/0b060fdbd92cbcc66b383bcaa9ba5e5e624d7ee3>`_
    .. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
        dimensionality reduction via tangent space alignment.
        Journal of Shanghai Univ.  8:406 (2004)

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import LocallyLinearEmbedding
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = LocallyLinearEmbedding(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    """
    _parameter_constraints: dict = {'n_neighbors': [Interval(Integral, 1, None, closed='left')], 'n_components': [Interval(Integral, 1, None, closed='left')], 'reg': [Interval(Real, 0, None, closed='left')], 'eigen_solver': [StrOptions({'auto', 'arpack', 'dense'})], 'tol': [Interval(Real, 0, None, closed='left')], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'method': [StrOptions({'standard', 'hessian', 'modified', 'ltsa'})], 'hessian_tol': [Interval(Real, 0, None, closed='left')], 'modified_tol': [Interval(Real, 0, None, closed='left')], 'neighbors_algorithm': [StrOptions({'auto', 'brute', 'kd_tree', 'ball_tree'})], 'random_state': ['random_state'], 'n_jobs': [None, Integral]}

    def __init__(self, *, n_neighbors=5, n_components=2, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None, n_jobs=None):
        if False:
            return 10
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.random_state = random_state
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs

    def _fit_transform(self, X):
        if False:
            return 10
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.neighbors_algorithm, n_jobs=self.n_jobs)
        random_state = check_random_state(self.random_state)
        X = self._validate_data(X, dtype=float)
        self.nbrs_.fit(X)
        (self.embedding_, self.reconstruction_error_) = locally_linear_embedding(X=self.nbrs_, n_neighbors=self.n_neighbors, n_components=self.n_components, eigen_solver=self.eigen_solver, tol=self.tol, max_iter=self.max_iter, method=self.method, hessian_tol=self.hessian_tol, modified_tol=self.modified_tol, random_state=random_state, reg=self.reg, n_jobs=self.n_jobs)
        self._n_features_out = self.embedding_.shape[1]

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        'Compute the embedding vectors for data X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training set.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        Returns\n        -------\n        self : object\n            Fitted `LocallyLinearEmbedding` class instance.\n        '
        self._fit_transform(X)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        'Compute the embedding vectors for data X and transform X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training set.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        Returns\n        -------\n        X_new : array-like, shape (n_samples, n_components)\n            Returns the instance itself.\n        '
        self._fit_transform(X)
        return self.embedding_

    def transform(self, X):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transform new points into embedding space.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training set.\n\n        Returns\n        -------\n        X_new : ndarray of shape (n_samples, n_components)\n            Returns the instance itself.\n\n        Notes\n        -----\n        Because of scaling performed by this method, it is discouraged to use\n        it together with methods that are not scale-invariant (like SVMs).\n        '
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        ind = self.nbrs_.kneighbors(X, n_neighbors=self.n_neighbors, return_distance=False)
        weights = barycenter_weights(X, self.nbrs_._fit_X, ind, reg=self.reg)
        X_new = np.empty((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            X_new[i] = np.dot(self.embedding_[ind[i]].T, weights[i])
        return X_new