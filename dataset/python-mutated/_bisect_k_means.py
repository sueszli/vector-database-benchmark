"""Bisecting K-means clustering."""
import warnings
import numpy as np
import scipy.sparse as sp
from ..base import _fit_context
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import StrOptions
from ..utils.extmath import row_norms
from ..utils.validation import _check_sample_weight, check_is_fitted, check_random_state
from ._k_means_common import _inertia_dense, _inertia_sparse
from ._kmeans import _BaseKMeans, _kmeans_single_elkan, _kmeans_single_lloyd, _labels_inertia_threadpool_limit

class _BisectingTree:
    """Tree structure representing the hierarchical clusters of BisectingKMeans."""

    def __init__(self, center, indices, score):
        if False:
            for i in range(10):
                print('nop')
        'Create a new cluster node in the tree.\n\n        The node holds the center of this cluster and the indices of the data points\n        that belong to it.\n        '
        self.center = center
        self.indices = indices
        self.score = score
        self.left = None
        self.right = None

    def split(self, labels, centers, scores):
        if False:
            print('Hello World!')
        'Split the cluster node into two subclusters.'
        self.left = _BisectingTree(indices=self.indices[labels == 0], center=centers[0], score=scores[0])
        self.right = _BisectingTree(indices=self.indices[labels == 1], center=centers[1], score=scores[1])
        self.indices = None

    def get_cluster_to_bisect(self):
        if False:
            return 10
        "Return the cluster node to bisect next.\n\n        It's based on the score of the cluster, which can be either the number of\n        data points assigned to that cluster or the inertia of that cluster\n        (see `bisecting_strategy` for details).\n        "
        max_score = None
        for cluster_leaf in self.iter_leaves():
            if max_score is None or cluster_leaf.score > max_score:
                max_score = cluster_leaf.score
                best_cluster_leaf = cluster_leaf
        return best_cluster_leaf

    def iter_leaves(self):
        if False:
            while True:
                i = 10
        'Iterate over all the cluster leaves in the tree.'
        if self.left is None:
            yield self
        else:
            yield from self.left.iter_leaves()
            yield from self.right.iter_leaves()

class BisectingKMeans(_BaseKMeans):
    """Bisecting K-Means clustering.

    Read more in the :ref:`User Guide <bisect_k_means>`.

    .. versionadded:: 1.1

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'} or callable, default='random'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : int, default=1
        Number of time the inner k-means algorithm will be run with different
        centroid seeds in each bisection.
        That will result producing for each bisection best output of n_init
        consecutive runs in terms of inertia.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization
        in inner K-Means. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    max_iter : int, default=300
        Maximum number of iterations of the inner k-means algorithm at each
        bisection.

    verbose : int, default=0
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations  to declare
        convergence. Used in inner k-means algorithm at each bisection to pick
        best possible clusters.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        Inner K-means algorithm used in bisection.
        The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

    bisecting_strategy : {"biggest_inertia", "largest_cluster"},            default="biggest_inertia"
        Defines how bisection should be performed:

         - "biggest_inertia" means that BisectingKMeans will always check
            all calculated cluster for cluster with biggest SSE
            (Sum of squared errors) and bisect it. This approach concentrates on
            precision, but may be costly in terms of execution time (especially for
            larger amount of data points).

         - "largest_cluster" - BisectingKMeans will always split cluster with
            largest amount of points assigned to it from all clusters
            previously calculated. That should work faster than picking by SSE
            ('biggest_inertia') and may produce similar results in most cases.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    KMeans : Original implementation of K-Means algorithm.

    Notes
    -----
    It might be inefficient when n_cluster is less than 3, due to unnecessary
    calculations for that case.

    Examples
    --------
    >>> from sklearn.cluster import BisectingKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 1], [10, 1], [3, 1],
    ...               [10, 0], [2, 1], [10, 2],
    ...               [10, 8], [10, 9], [10, 10]])
    >>> bisect_means = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
    >>> bisect_means.labels_
    array([0, 2, 0, 2, 0, 2, 1, 1, 1], dtype=int32)
    >>> bisect_means.predict([[0, 0], [12, 3]])
    array([0, 2], dtype=int32)
    >>> bisect_means.cluster_centers_
    array([[ 2., 1.],
           [10., 9.],
           [10., 1.]])
    """
    _parameter_constraints: dict = {**_BaseKMeans._parameter_constraints, 'init': [StrOptions({'k-means++', 'random'}), callable], 'copy_x': ['boolean'], 'algorithm': [StrOptions({'lloyd', 'elkan'})], 'bisecting_strategy': [StrOptions({'biggest_inertia', 'largest_cluster'})]}

    def __init__(self, n_clusters=8, *, init='random', n_init=1, random_state=None, max_iter=300, verbose=0, tol=0.0001, copy_x=True, algorithm='lloyd', bisecting_strategy='biggest_inertia'):
        if False:
            print('Hello World!')
        super().__init__(n_clusters=n_clusters, init=init, max_iter=max_iter, verbose=verbose, random_state=random_state, tol=tol, n_init=n_init)
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.bisecting_strategy = bisecting_strategy

    def _warn_mkl_vcomp(self, n_active_threads):
        if False:
            return 10
        'Warn when vcomp and mkl are both present'
        warnings.warn(f'BisectingKMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS={n_active_threads}.')

    def _inertia_per_cluster(self, X, centers, labels, sample_weight):
        if False:
            print('Hello World!')
        'Calculate the sum of squared errors (inertia) per cluster.\n\n        Parameters\n        ----------\n        X : {ndarray, csr_matrix} of shape (n_samples, n_features)\n            The input samples.\n\n        centers : ndarray of shape (n_clusters=2, n_features)\n            The cluster centers.\n\n        labels : ndarray of shape (n_samples,)\n            Index of the cluster each sample belongs to.\n\n        sample_weight : ndarray of shape (n_samples,)\n            The weights for each observation in X.\n\n        Returns\n        -------\n        inertia_per_cluster : ndarray of shape (n_clusters=2,)\n            Sum of squared errors (inertia) for each cluster.\n        '
        n_clusters = centers.shape[0]
        _inertia = _inertia_sparse if sp.issparse(X) else _inertia_dense
        inertia_per_cluster = np.empty(n_clusters)
        for label in range(n_clusters):
            inertia_per_cluster[label] = _inertia(X, sample_weight, centers, labels, self._n_threads, single_label=label)
        return inertia_per_cluster

    def _bisect(self, X, x_squared_norms, sample_weight, cluster_to_bisect):
        if False:
            for i in range(10):
                print('nop')
        'Split a cluster into 2 subsclusters.\n\n        Parameters\n        ----------\n        X : {ndarray, csr_matrix} of shape (n_samples, n_features)\n            Training instances to cluster.\n\n        x_squared_norms : ndarray of shape (n_samples,)\n            Squared euclidean norm of each data point.\n\n        sample_weight : ndarray of shape (n_samples,)\n            The weights for each observation in X.\n\n        cluster_to_bisect : _BisectingTree node object\n            The cluster node to split.\n        '
        X = X[cluster_to_bisect.indices]
        x_squared_norms = x_squared_norms[cluster_to_bisect.indices]
        sample_weight = sample_weight[cluster_to_bisect.indices]
        best_inertia = None
        for _ in range(self.n_init):
            centers_init = self._init_centroids(X, x_squared_norms=x_squared_norms, init=self.init, random_state=self._random_state, n_centroids=2, sample_weight=sample_weight)
            (labels, inertia, centers, _) = self._kmeans_single(X, sample_weight, centers_init, max_iter=self.max_iter, verbose=self.verbose, tol=self.tol, n_threads=self._n_threads)
            if best_inertia is None or inertia < best_inertia * (1 - 1e-06):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
        if self.verbose:
            print(f'New centroids from bisection: {best_centers}')
        if self.bisecting_strategy == 'biggest_inertia':
            scores = self._inertia_per_cluster(X, best_centers, best_labels, sample_weight)
        else:
            scores = np.bincount(best_labels, minlength=2)
        cluster_to_bisect.split(best_labels, best_centers, scores)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        if False:
            i = 10
            return i + 15
        'Compute bisecting k-means clustering.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n\n            Training instances to cluster.\n\n            .. note:: The data will be converted to C ordering,\n                which will cause a memory copy\n                if the given data is not C-contiguous.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            The weights for each observation in X. If None, all observations\n            are assigned equal weight. `sample_weight` is not used during\n            initialization if `init` is a callable.\n\n        Returns\n        -------\n        self\n            Fitted estimator.\n        '
        X = self._validate_data(X, accept_sparse='csr', dtype=[np.float64, np.float32], order='C', copy=self.copy_x, accept_large_sparse=False)
        self._check_params_vs_input(X)
        self._random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()
        if self.algorithm == 'lloyd' or self.n_clusters == 1:
            self._kmeans_single = _kmeans_single_lloyd
            self._check_mkl_vcomp(X, X.shape[0])
        else:
            self._kmeans_single = _kmeans_single_elkan
        if not sp.issparse(X):
            self._X_mean = X.mean(axis=0)
            X -= self._X_mean
        self._bisecting_tree = _BisectingTree(indices=np.arange(X.shape[0]), center=X.mean(axis=0), score=0)
        x_squared_norms = row_norms(X, squared=True)
        for _ in range(self.n_clusters - 1):
            cluster_to_bisect = self._bisecting_tree.get_cluster_to_bisect()
            self._bisect(X, x_squared_norms, sample_weight, cluster_to_bisect)
        self.labels_ = np.full(X.shape[0], -1, dtype=np.int32)
        self.cluster_centers_ = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)
        for (i, cluster_node) in enumerate(self._bisecting_tree.iter_leaves()):
            self.labels_[cluster_node.indices] = i
            self.cluster_centers_[i] = cluster_node.center
            cluster_node.label = i
            cluster_node.indices = None
        if not sp.issparse(X):
            X += self._X_mean
            self.cluster_centers_ += self._X_mean
        _inertia = _inertia_sparse if sp.issparse(X) else _inertia_dense
        self.inertia_ = _inertia(X, sample_weight, self.cluster_centers_, self.labels_, self._n_threads)
        self._n_features_out = self.cluster_centers_.shape[0]
        return self

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        'Predict which cluster each sample in X belongs to.\n\n        Prediction is made by going down the hierarchical tree\n        in searching of closest leaf cluster.\n\n        In the vector quantization literature, `cluster_centers_` is called\n        the code book and each value returned by `predict` is the index of\n        the closest code in the code book.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            New data to predict.\n\n        Returns\n        -------\n        labels : ndarray of shape (n_samples,)\n            Index of the cluster each sample belongs to.\n        '
        check_is_fitted(self)
        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = np.ones_like(x_squared_norms)
        labels = self._predict_recursive(X, sample_weight, self._bisecting_tree)
        return labels

    def _predict_recursive(self, X, sample_weight, cluster_node):
        if False:
            print('Hello World!')
        'Predict recursively by going down the hierarchical tree.\n\n        Parameters\n        ----------\n        X : {ndarray, csr_matrix} of shape (n_samples, n_features)\n            The data points, currently assigned to `cluster_node`, to predict between\n            the subclusters of this node.\n\n        sample_weight : ndarray of shape (n_samples,)\n            The weights for each observation in X.\n\n        cluster_node : _BisectingTree node object\n            The cluster node of the hierarchical tree.\n\n        Returns\n        -------\n        labels : ndarray of shape (n_samples,)\n            Index of the cluster each sample belongs to.\n        '
        if cluster_node.left is None:
            return np.full(X.shape[0], cluster_node.label, dtype=np.int32)
        centers = np.vstack((cluster_node.left.center, cluster_node.right.center))
        if hasattr(self, '_X_mean'):
            centers += self._X_mean
        cluster_labels = _labels_inertia_threadpool_limit(X, sample_weight, centers, self._n_threads, return_inertia=False)
        mask = cluster_labels == 0
        labels = np.full(X.shape[0], -1, dtype=np.int32)
        labels[mask] = self._predict_recursive(X[mask], sample_weight[mask], cluster_node.left)
        labels[~mask] = self._predict_recursive(X[~mask], sample_weight[~mask], cluster_node.right)
        return labels

    def _more_tags(self):
        if False:
            for i in range(10):
                print('nop')
        return {'preserves_dtype': [np.float64, np.float32]}