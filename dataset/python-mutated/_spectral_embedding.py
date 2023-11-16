"""Spectral Embedding."""
import warnings
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, lobpcg
from ..base import BaseEstimator, _fit_context
from ..metrics.pairwise import rbf_kernel
from ..neighbors import NearestNeighbors, kneighbors_graph
from ..utils import check_array, check_random_state, check_symmetric
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import _deterministic_vector_sign_flip
from ..utils.fixes import laplacian as csgraph_laplacian
from ..utils.fixes import parse_version, sp_version

def _graph_connected_component(graph, node_id):
    if False:
        while True:
            i = 10
    'Find the largest graph connected components that contains one\n    given node.\n\n    Parameters\n    ----------\n    graph : array-like of shape (n_samples, n_samples)\n        Adjacency matrix of the graph, non-zero weight means an edge\n        between the nodes.\n\n    node_id : int\n        The index of the query node of the graph.\n\n    Returns\n    -------\n    connected_components_matrix : array-like of shape (n_samples,)\n        An array of bool value indicating the indexes of the nodes\n        belonging to the largest connected components of the given query\n        node.\n    '
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=bool)
    nodes_to_explore = np.zeros(n_node, dtype=bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[[i], :].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes

def _graph_is_connected(graph):
    if False:
        i = 10
        return i + 15
    'Return whether the graph is connected (True) or Not (False).\n\n    Parameters\n    ----------\n    graph : {array-like, sparse matrix} of shape (n_samples, n_samples)\n        Adjacency matrix of the graph, non-zero weight means an edge\n        between the nodes.\n\n    Returns\n    -------\n    is_connected : bool\n        True means the graph is fully connected and False means not.\n    '
    if sparse.issparse(graph):
        accept_large_sparse = sp_version >= parse_version('1.11.3')
        graph = check_array(graph, accept_sparse=True, accept_large_sparse=accept_large_sparse)
        (n_connected_components, _) = connected_components(graph)
        return n_connected_components == 1
    else:
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]

def _set_diag(laplacian, value, norm_laplacian):
    if False:
        print('Hello World!')
    'Set the diagonal of the laplacian matrix and convert it to a\n    sparse format well suited for eigenvalue decomposition.\n\n    Parameters\n    ----------\n    laplacian : {ndarray, sparse matrix}\n        The graph laplacian.\n\n    value : float\n        The value of the diagonal.\n\n    norm_laplacian : bool\n        Whether the value of the diagonal should be changed or not.\n\n    Returns\n    -------\n    laplacian : {array, sparse matrix}\n        An array of matrix in a form that is well suited to fast\n        eigenvalue decomposition, depending on the band width of the\n        matrix.\n    '
    n_nodes = laplacian.shape[0]
    if not sparse.issparse(laplacian):
        if norm_laplacian:
            laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            laplacian = laplacian.todia()
        else:
            laplacian = laplacian.tocsr()
    return laplacian

def spectral_embedding(adjacency, *, n_components=8, eigen_solver=None, random_state=None, eigen_tol='auto', norm_laplacian=True, drop_first=True):
    if False:
        return 10
    'Project the sample on the first eigenvectors of the graph Laplacian.\n\n    The adjacency matrix is used to compute a normalized graph Laplacian\n    whose spectrum (especially the eigenvectors associated to the\n    smallest eigenvalues) has an interpretation in terms of minimal\n    number of cuts necessary to split the graph into comparably sized\n    components.\n\n    This embedding can also \'work\' even if the ``adjacency`` variable is\n    not strictly the adjacency matrix of a graph but more generally\n    an affinity or similarity matrix between samples (for instance the\n    heat kernel of a euclidean distance matrix or a k-NN matrix).\n\n    However care must taken to always make the affinity matrix symmetric\n    so that the eigenvector decomposition works as expected.\n\n    Note : Laplacian Eigenmaps is the actual algorithm implemented here.\n\n    Read more in the :ref:`User Guide <spectral_embedding>`.\n\n    Parameters\n    ----------\n    adjacency : {array-like, sparse graph} of shape (n_samples, n_samples)\n        The adjacency matrix of the graph to embed.\n\n    n_components : int, default=8\n        The dimension of the projection subspace.\n\n    eigen_solver : {\'arpack\', \'lobpcg\', \'amg\'}, default=None\n        The eigenvalue decomposition strategy to use. AMG requires pyamg\n        to be installed. It can be faster on very large, sparse problems,\n        but may also lead to instabilities. If None, then ``\'arpack\'`` is\n        used.\n\n    random_state : int, RandomState instance or None, default=None\n        A pseudo random number generator used for the initialization\n        of the lobpcg eigen vectors decomposition when `eigen_solver ==\n        \'amg\'`, and for the K-Means initialization. Use an int to make\n        the results deterministic across calls (See\n        :term:`Glossary <random_state>`).\n\n        .. note::\n            When using `eigen_solver == \'amg\'`,\n            it is necessary to also fix the global numpy seed with\n            `np.random.seed(int)` to get deterministic results. See\n            https://github.com/pyamg/pyamg/issues/139 for further\n            information.\n\n    eigen_tol : float, default="auto"\n        Stopping criterion for eigendecomposition of the Laplacian matrix.\n        If `eigen_tol="auto"` then the passed tolerance will depend on the\n        `eigen_solver`:\n\n        - If `eigen_solver="arpack"`, then `eigen_tol=0.0`;\n        - If `eigen_solver="lobpcg"` or `eigen_solver="amg"`, then\n          `eigen_tol=None` which configures the underlying `lobpcg` solver to\n          automatically resolve the value according to their heuristics. See,\n          :func:`scipy.sparse.linalg.lobpcg` for details.\n\n        Note that when using `eigen_solver="amg"` values of `tol<1e-5` may lead\n        to convergence issues and should be avoided.\n\n        .. versionadded:: 1.2\n           Added \'auto\' option.\n\n    norm_laplacian : bool, default=True\n        If True, then compute symmetric normalized Laplacian.\n\n    drop_first : bool, default=True\n        Whether to drop the first eigenvector. For spectral embedding, this\n        should be True as the first eigenvector should be constant vector for\n        connected graph, but for spectral clustering, this should be kept as\n        False to retain the first eigenvector.\n\n    Returns\n    -------\n    embedding : ndarray of shape (n_samples, n_components)\n        The reduced samples.\n\n    Notes\n    -----\n    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph\n    has one connected component. If there graph has many components, the first\n    few eigenvectors will simply uncover the connected components of the graph.\n\n    References\n    ----------\n    * https://en.wikipedia.org/wiki/LOBPCG\n\n    * :doi:`"Toward the Optimal Preconditioned Eigensolver: Locally Optimal\n      Block Preconditioned Conjugate Gradient Method",\n      Andrew V. Knyazev\n      <10.1137/S1064827500366124>`\n    '
    adjacency = check_symmetric(adjacency)
    if eigen_solver == 'amg':
        try:
            from pyamg import smoothed_aggregation_solver
        except ImportError as e:
            raise ValueError("The eigen_solver was set to 'amg', but pyamg is not available.") from e
    if eigen_solver is None:
        eigen_solver = 'arpack'
    elif eigen_solver not in ('arpack', 'lobpcg', 'amg'):
        raise ValueError("Unknown value for eigen_solver: '%s'.Should be 'amg', 'arpack', or 'lobpcg'" % eigen_solver)
    random_state = check_random_state(random_state)
    n_nodes = adjacency.shape[0]
    if drop_first:
        n_components = n_components + 1
    if not _graph_is_connected(adjacency):
        warnings.warn('Graph is not fully connected, spectral embedding may not work as expected.')
    (laplacian, dd) = csgraph_laplacian(adjacency, normed=norm_laplacian, return_diag=True)
    if eigen_solver == 'arpack' or (eigen_solver != 'lobpcg' and (not sparse.issparse(laplacian) or n_nodes < 5 * n_components)):
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        try:
            tol = 0 if eigen_tol == 'auto' else eigen_tol
            laplacian *= -1
            v0 = _init_arpack_v0(laplacian.shape[0], random_state)
            laplacian = check_array(laplacian, accept_sparse='csr', accept_large_sparse=False)
            (_, diffusion_map) = eigsh(laplacian, k=n_components, sigma=1.0, which='LM', tol=tol, v0=v0)
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                embedding = embedding / dd
        except RuntimeError:
            eigen_solver = 'lobpcg'
            laplacian *= -1
    elif eigen_solver == 'amg':
        if not sparse.issparse(laplacian):
            warnings.warn('AMG works better for sparse matrices')
        laplacian = check_array(laplacian, dtype=[np.float64, np.float32], accept_sparse=True)
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        diag_shift = 1e-05 * sparse.eye(laplacian.shape[0])
        laplacian += diag_shift
        if hasattr(sparse, 'csr_array') and isinstance(laplacian, sparse.csr_array):
            laplacian = sparse.csr_matrix(laplacian)
        ml = smoothed_aggregation_solver(check_array(laplacian, accept_sparse='csr'))
        laplacian -= diag_shift
        M = ml.aspreconditioner()
        X = random_state.standard_normal(size=(laplacian.shape[0], n_components + 1))
        X[:, 0] = dd.ravel()
        X = X.astype(laplacian.dtype)
        tol = None if eigen_tol == 'auto' else eigen_tol
        (_, diffusion_map) = lobpcg(laplacian, X, M=M, tol=tol, largest=False)
        embedding = diffusion_map.T
        if norm_laplacian:
            embedding = embedding / dd
        if embedding.shape[0] == 1:
            raise ValueError
    if eigen_solver == 'lobpcg':
        laplacian = check_array(laplacian, dtype=[np.float64, np.float32], accept_sparse=True)
        if n_nodes < 5 * n_components + 1:
            if sparse.issparse(laplacian):
                laplacian = laplacian.toarray()
            (_, diffusion_map) = eigh(laplacian, check_finite=False)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd
        else:
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            X = random_state.standard_normal(size=(laplacian.shape[0], n_components + 1))
            X[:, 0] = dd.ravel()
            X = X.astype(laplacian.dtype)
            tol = None if eigen_tol == 'auto' else eigen_tol
            (_, diffusion_map) = lobpcg(laplacian, X, tol=tol, largest=False, maxiter=2000)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd
            if embedding.shape[0] == 1:
                raise ValueError
    embedding = _deterministic_vector_sign_flip(embedding)
    if drop_first:
        return embedding[1:n_components].T
    else:
        return embedding[:n_components].T

class SpectralEmbedding(BaseEstimator):
    """Spectral embedding for non-linear dimensionality reduction.

    Forms an affinity matrix given by the specified function and
    applies spectral decomposition to the corresponding graph laplacian.
    The resulting transformation is given by the value of the
    eigenvectors for each data point.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Read more in the :ref:`User Guide <spectral_embedding>`.

    Parameters
    ----------
    n_components : int, default=2
        The dimension of the projected subspace.

    affinity : {'nearest_neighbors', 'rbf', 'precomputed',                 'precomputed_nearest_neighbors'} or callable,                 default='nearest_neighbors'
        How to construct the affinity matrix.
         - 'nearest_neighbors' : construct the affinity matrix by computing a
           graph of nearest neighbors.
         - 'rbf' : construct the affinity matrix by computing a radial basis
           function (RBF) kernel.
         - 'precomputed' : interpret ``X`` as a precomputed affinity matrix.
         - 'precomputed_nearest_neighbors' : interpret ``X`` as a sparse graph
           of precomputed nearest neighbors, and constructs the affinity matrix
           by selecting the ``n_neighbors`` nearest neighbors.
         - callable : use passed in function as affinity
           the function takes in data matrix (n_samples, n_features)
           and return affinity matrix (n_samples, n_samples).

    gamma : float, default=None
        Kernel coefficient for rbf kernel. If None, gamma will be set to
        1/n_features.

    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigen vectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.

    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems.
        If None, then ``'arpack'`` is used.

    eigen_tol : float, default="auto"
        Stopping criterion for eigendecomposition of the Laplacian matrix.
        If `eigen_tol="auto"` then the passed tolerance will depend on the
        `eigen_solver`:

        - If `eigen_solver="arpack"`, then `eigen_tol=0.0`;
        - If `eigen_solver="lobpcg"` or `eigen_solver="amg"`, then
          `eigen_tol=None` which configures the underlying `lobpcg` solver to
          automatically resolve the value according to their heuristics. See,
          :func:`scipy.sparse.linalg.lobpcg` for details.

        Note that when using `eigen_solver="lobpcg"` or `eigen_solver="amg"`
        values of `tol<1e-5` may lead to convergence issues and should be
        avoided.

        .. versionadded:: 1.2

    n_neighbors : int, default=None
        Number of nearest neighbors for nearest_neighbors graph building.
        If None, n_neighbors will be set to max(n_samples/10, 1).

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Spectral embedding of the training matrix.

    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Affinity_matrix constructed from samples or precomputed.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_neighbors_ : int
        Number of nearest neighbors effectively used.

    See Also
    --------
    Isomap : Non-linear dimensionality reduction through Isometric Mapping.

    References
    ----------

    - :doi:`A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      <10.1007/s11222-007-9033-z>`

    - `On Spectral Clustering: Analysis and an algorithm, 2001
      Andrew Y. Ng, Michael I. Jordan, Yair Weiss
      <https://citeseerx.ist.psu.edu/doc_view/pid/796c5d6336fc52aa84db575fb821c78918b65f58>`_

    - :doi:`Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      <10.1109/34.868688>`

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import SpectralEmbedding
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = SpectralEmbedding(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left')], 'affinity': [StrOptions({'nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'}), callable], 'gamma': [Interval(Real, 0, None, closed='left'), None], 'random_state': ['random_state'], 'eigen_solver': [StrOptions({'arpack', 'lobpcg', 'amg'}), None], 'eigen_tol': [Interval(Real, 0, None, closed='left'), StrOptions({'auto'})], 'n_neighbors': [Interval(Integral, 1, None, closed='left'), None], 'n_jobs': [None, Integral]}

    def __init__(self, n_components=2, *, affinity='nearest_neighbors', gamma=None, random_state=None, eigen_solver=None, eigen_tol='auto', n_neighbors=None, n_jobs=None):
        if False:
            for i in range(10):
                print('nop')
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.eigen_tol = eigen_tol
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def _more_tags(self):
        if False:
            i = 10
            return i + 15
        return {'pairwise': self.affinity in ['precomputed', 'precomputed_nearest_neighbors']}

    def _get_affinity_matrix(self, X, Y=None):
        if False:
            i = 10
            return i + 15
        'Calculate the affinity matrix from data\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training vector, where `n_samples` is the number of samples\n            and `n_features` is the number of features.\n\n            If affinity is "precomputed"\n            X : array-like of shape (n_samples, n_samples),\n            Interpret X as precomputed adjacency graph computed from\n            samples.\n\n        Y: Ignored\n\n        Returns\n        -------\n        affinity_matrix of shape (n_samples, n_samples)\n        '
        if self.affinity == 'precomputed':
            self.affinity_matrix_ = X
            return self.affinity_matrix_
        if self.affinity == 'precomputed_nearest_neighbors':
            estimator = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=self.n_jobs, metric='precomputed').fit(X)
            connectivity = estimator.kneighbors_graph(X=X, mode='connectivity')
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
            return self.affinity_matrix_
        if self.affinity == 'nearest_neighbors':
            if sparse.issparse(X):
                warnings.warn('Nearest neighbors affinity currently does not support sparse input, falling back to rbf affinity')
                self.affinity = 'rbf'
            else:
                self.n_neighbors_ = self.n_neighbors if self.n_neighbors is not None else max(int(X.shape[0] / 10), 1)
                self.affinity_matrix_ = kneighbors_graph(X, self.n_neighbors_, include_self=True, n_jobs=self.n_jobs)
                self.affinity_matrix_ = 0.5 * (self.affinity_matrix_ + self.affinity_matrix_.T)
                return self.affinity_matrix_
        if self.affinity == 'rbf':
            self.gamma_ = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            self.affinity_matrix_ = rbf_kernel(X, gamma=self.gamma_)
            return self.affinity_matrix_
        self.affinity_matrix_ = self.affinity(X)
        return self.affinity_matrix_

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        if False:
            i = 10
            return i + 15
        'Fit the model from data in X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training vector, where `n_samples` is the number of samples\n            and `n_features` is the number of features.\n\n            If affinity is "precomputed"\n            X : {array-like, sparse matrix}, shape (n_samples, n_samples),\n            Interpret X as precomputed adjacency graph computed from\n            samples.\n\n        y : Ignored\n            Not used, present for API consistency by convention.\n\n        Returns\n        -------\n        self : object\n            Returns the instance itself.\n        '
        X = self._validate_data(X, accept_sparse='csr', ensure_min_samples=2)
        random_state = check_random_state(self.random_state)
        affinity_matrix = self._get_affinity_matrix(X)
        self.embedding_ = spectral_embedding(affinity_matrix, n_components=self.n_components, eigen_solver=self.eigen_solver, eigen_tol=self.eigen_tol, random_state=random_state)
        return self

    def fit_transform(self, X, y=None):
        if False:
            while True:
                i = 10
        'Fit the model from data in X and transform X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training vector, where `n_samples` is the number of samples\n            and `n_features` is the number of features.\n\n            If affinity is "precomputed"\n            X : {array-like, sparse matrix} of shape (n_samples, n_samples),\n            Interpret X as precomputed adjacency graph computed from\n            samples.\n\n        y : Ignored\n            Not used, present for API consistency by convention.\n\n        Returns\n        -------\n        X_new : array-like of shape (n_samples, n_components)\n            Spectral embedding of the training matrix.\n        '
        self.fit(X)
        return self.embedding_