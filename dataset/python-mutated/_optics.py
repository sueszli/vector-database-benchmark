"""Ordering Points To Identify the Clustering Structure (OPTICS)

These routines execute the OPTICS algorithm, and implement various
cluster extraction methods of the ordered list.

Authors: Shane Grigsby <refuge@rocktalus.com>
         Adrin Jalali <adrinjalali@gmail.com>
         Erich Schubert <erich@debian.org>
         Hanmin Qin <qinhanmin2005@sina.com>
License: BSD 3 clause
"""
import warnings
from numbers import Integral, Real
import numpy as np
from scipy.sparse import SparseEfficiencyWarning, issparse
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..exceptions import DataConversionWarning
from ..metrics import pairwise_distances
from ..metrics.pairwise import _VALID_METRICS, PAIRWISE_BOOLEAN_FUNCTIONS
from ..neighbors import NearestNeighbors
from ..utils import gen_batches, get_chunk_n_rows
from ..utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions, validate_params
from ..utils.validation import check_memory

class OPTICS(ClusterMixin, BaseEstimator):
    """Estimate clustering structure from vector array.

    OPTICS (Ordering Points To Identify the Clustering Structure), closely
    related to DBSCAN, finds core sample of high density and expands clusters
    from them [1]_. Unlike DBSCAN, keeps cluster hierarchy for a variable
    neighborhood radius. Better suited for usage on large datasets than the
    current sklearn implementation of DBSCAN.

    Clusters are then extracted using a DBSCAN-like method
    (cluster_method = 'dbscan') or an automatic
    technique proposed in [1]_ (cluster_method = 'xi').

    This implementation deviates from the original OPTICS by first performing
    k-nearest-neighborhood searches on all points to identify core sizes, then
    computing only the distances to unprocessed points when constructing the
    cluster order. Note that we do not employ a heap to manage the expansion
    candidates, so the time complexity will be O(n^2).

    Read more in the :ref:`User Guide <optics>`.

    Parameters
    ----------
    min_samples : int > 1 or float between 0 and 1, default=5
        The number of samples in a neighborhood for a point to be considered as
        a core point. Also, up and down steep regions can't have more than
        ``min_samples`` consecutive non-steep points. Expressed as an absolute
        number or a fraction of the number of samples (rounded to be at least
        2).

    max_eps : float, default=np.inf
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Default value of ``np.inf`` will
        identify clusters across all scales; reducing ``max_eps`` will result
        in shorter run times.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string. If metric is
        "precomputed", `X` is assumed to be a distance matrix and must be
        square.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        Sparse matrices are only supported by scikit-learn metrics.
        See the documentation for scipy.spatial.distance for details on these
        metrics.

        .. note::
           `'kulsinski'` is deprecated from SciPy 1.9 and will removed in SciPy 1.11.

    p : float, default=2
        Parameter for the Minkowski metric from
        :class:`~sklearn.metrics.pairwise_distances`. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    cluster_method : str, default='xi'
        The extraction method used to extract clusters using the calculated
        reachability and ordering. Possible values are "xi" and "dbscan".

    eps : float, default=None
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. By default it assumes the same value
        as ``max_eps``.
        Used only when ``cluster_method='dbscan'``.

    xi : float between 0 and 1, default=0.05
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.
        Used only when ``cluster_method='xi'``.

    predecessor_correction : bool, default=True
        Correct clusters according to the predecessors calculated by OPTICS
        [2]_. This parameter has minimal effect on most datasets.
        Used only when ``cluster_method='xi'``.

    min_cluster_size : int > 1 or float between 0 and 1, default=None
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded to be
        at least 2). If ``None``, the value of ``min_samples`` is used instead.
        Used only when ``cluster_method='xi'``.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`~sklearn.neighbors.BallTree`.
        - 'kd_tree' will use :class:`~sklearn.neighbors.KDTree`.
        - 'brute' will use a brute-force search.
        - 'auto' (default) will attempt to decide the most appropriate
          algorithm based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to :class:`~sklearn.neighbors.BallTree` or
        :class:`~sklearn.neighbors.KDTree`. This can affect the speed of the
        construction and query, as well as the memory required to store the
        tree. The optimal value depends on the nature of the problem.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples and points which are not included in a leaf cluster
        of ``cluster_hierarchy_`` are labeled as -1.

    reachability_ : ndarray of shape (n_samples,)
        Reachability distances per sample, indexed by object order. Use
        ``clust.reachability_[clust.ordering_]`` to access in cluster order.

    ordering_ : ndarray of shape (n_samples,)
        The cluster ordered list of sample indices.

    core_distances_ : ndarray of shape (n_samples,)
        Distance at which each sample becomes a core point, indexed by object
        order. Points which will never be core have a distance of inf. Use
        ``clust.core_distances_[clust.ordering_]`` to access in cluster order.

    predecessor_ : ndarray of shape (n_samples,)
        Point that a sample was reached from, indexed by object order.
        Seed points have a predecessor of -1.

    cluster_hierarchy_ : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of ``[start, end]`` in each row, with
        all indices inclusive. The clusters are ordered according to
        ``(end, -start)`` (ascending) so that larger clusters encompassing
        smaller clusters come after those smaller ones. Since ``labels_`` does
        not reflect the hierarchy, usually
        ``len(cluster_hierarchy_) > np.unique(optics.labels_)``. Please also
        note that these indices are of the ``ordering_``, i.e.
        ``X[ordering_][start:end + 1]`` form a cluster.
        Only available when ``cluster_method='xi'``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    DBSCAN : A similar clustering for a specified neighborhood radius (eps).
        Our implementation is optimized for runtime.

    References
    ----------
    .. [1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel,
       and Jörg Sander. "OPTICS: ordering points to identify the clustering
       structure." ACM SIGMOD Record 28, no. 2 (1999): 49-60.

    .. [2] Schubert, Erich, Michael Gertz.
       "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of
       the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.

    Examples
    --------
    >>> from sklearn.cluster import OPTICS
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    >>> clustering = OPTICS(min_samples=2).fit(X)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])

    For a more detailed example see
    :ref:`sphx_glr_auto_examples_cluster_plot_optics.py`.
    """
    _parameter_constraints: dict = {'min_samples': [Interval(Integral, 2, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both')], 'max_eps': [Interval(Real, 0, None, closed='both')], 'metric': [StrOptions(set(_VALID_METRICS) | {'precomputed'}), callable], 'p': [Interval(Real, 1, None, closed='left')], 'metric_params': [dict, None], 'cluster_method': [StrOptions({'dbscan', 'xi'})], 'eps': [Interval(Real, 0, None, closed='both'), None], 'xi': [Interval(Real, 0, 1, closed='both')], 'predecessor_correction': ['boolean'], 'min_cluster_size': [Interval(Integral, 2, None, closed='left'), Interval(RealNotInt, 0, 1, closed='right'), None], 'algorithm': [StrOptions({'auto', 'brute', 'ball_tree', 'kd_tree'})], 'leaf_size': [Interval(Integral, 1, None, closed='left')], 'memory': [str, HasMethods('cache'), None], 'n_jobs': [Integral, None]}

    def __init__(self, *, min_samples=5, max_eps=np.inf, metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, memory=None, n_jobs=None):
        if False:
            i = 10
            return i + 15
        self.max_eps = max_eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.leaf_size = leaf_size
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.memory = memory
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None):
        if False:
            return 10
        "Perform OPTICS clustering.\n\n        Extracts an ordered list of points and reachability distances, and\n        performs initial clustering using ``max_eps`` distance specified at\n        OPTICS object instantiation.\n\n        Parameters\n        ----------\n        X : {ndarray, sparse matrix} of shape (n_samples, n_features), or                 (n_samples, n_samples) if metric='precomputed'\n            A feature array, or array of distances between samples if\n            metric='precomputed'. If a sparse matrix is provided, it will be\n            converted into CSR format.\n\n        y : Ignored\n            Not used, present for API consistency by convention.\n\n        Returns\n        -------\n        self : object\n            Returns a fitted instance of self.\n        "
        dtype = bool if self.metric in PAIRWISE_BOOLEAN_FUNCTIONS else float
        if dtype == bool and X.dtype != bool:
            msg = f'Data will be converted to boolean for metric {self.metric}, to avoid this warning, you may convert the data prior to calling fit.'
            warnings.warn(msg, DataConversionWarning)
        X = self._validate_data(X, dtype=dtype, accept_sparse='csr')
        if self.metric == 'precomputed' and issparse(X):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', SparseEfficiencyWarning)
                X.setdiag(X.diagonal())
        memory = check_memory(self.memory)
        (self.ordering_, self.core_distances_, self.reachability_, self.predecessor_) = memory.cache(compute_optics_graph)(X=X, min_samples=self.min_samples, algorithm=self.algorithm, leaf_size=self.leaf_size, metric=self.metric, metric_params=self.metric_params, p=self.p, n_jobs=self.n_jobs, max_eps=self.max_eps)
        if self.cluster_method == 'xi':
            (labels_, clusters_) = cluster_optics_xi(reachability=self.reachability_, predecessor=self.predecessor_, ordering=self.ordering_, min_samples=self.min_samples, min_cluster_size=self.min_cluster_size, xi=self.xi, predecessor_correction=self.predecessor_correction)
            self.cluster_hierarchy_ = clusters_
        elif self.cluster_method == 'dbscan':
            if self.eps is None:
                eps = self.max_eps
            else:
                eps = self.eps
            if eps > self.max_eps:
                raise ValueError('Specify an epsilon smaller than %s. Got %s.' % (self.max_eps, eps))
            labels_ = cluster_optics_dbscan(reachability=self.reachability_, core_distances=self.core_distances_, ordering=self.ordering_, eps=eps)
        self.labels_ = labels_
        return self

def _validate_size(size, n_samples, param_name):
    if False:
        for i in range(10):
            print('nop')
    if size > n_samples:
        raise ValueError('%s must be no greater than the number of samples (%d). Got %d' % (param_name, n_samples, size))

def _compute_core_distances_(X, neighbors, min_samples, working_memory):
    if False:
        print('Hello World!')
    "Compute the k-th nearest neighbor of each sample.\n\n    Equivalent to neighbors.kneighbors(X, self.min_samples)[0][:, -1]\n    but with more memory efficiency.\n\n    Parameters\n    ----------\n    X : array-like of shape (n_samples, n_features)\n        The data.\n    neighbors : NearestNeighbors instance\n        The fitted nearest neighbors estimator.\n    working_memory : int, default=None\n        The sought maximum memory for temporary distance matrix chunks.\n        When None (default), the value of\n        ``sklearn.get_config()['working_memory']`` is used.\n\n    Returns\n    -------\n    core_distances : ndarray of shape (n_samples,)\n        Distance at which each sample becomes a core point.\n        Points which will never be core have a distance of inf.\n    "
    n_samples = X.shape[0]
    core_distances = np.empty(n_samples)
    core_distances.fill(np.nan)
    chunk_n_rows = get_chunk_n_rows(row_bytes=16 * min_samples, max_n_rows=n_samples, working_memory=working_memory)
    slices = gen_batches(n_samples, chunk_n_rows)
    for sl in slices:
        core_distances[sl] = neighbors.kneighbors(X[sl], min_samples)[0][:, -1]
    return core_distances

@validate_params({'X': [np.ndarray, 'sparse matrix'], 'min_samples': [Interval(Integral, 2, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both')], 'max_eps': [Interval(Real, 0, None, closed='both')], 'metric': [StrOptions(set(_VALID_METRICS) | {'precomputed'}), callable], 'p': [Interval(Real, 0, None, closed='right'), None], 'metric_params': [dict, None], 'algorithm': [StrOptions({'auto', 'brute', 'ball_tree', 'kd_tree'})], 'leaf_size': [Interval(Integral, 1, None, closed='left')], 'n_jobs': [Integral, None]}, prefer_skip_nested_validation=False)
def compute_optics_graph(X, *, min_samples, max_eps, metric, p, metric_params, algorithm, leaf_size, n_jobs):
    if False:
        print('Hello World!')
    'Compute the OPTICS reachability graph.\n\n    Read more in the :ref:`User Guide <optics>`.\n\n    Parameters\n    ----------\n    X : {ndarray, sparse matrix} of shape (n_samples, n_features), or             (n_samples, n_samples) if metric=\'precomputed\'\n        A feature array, or array of distances between samples if\n        metric=\'precomputed\'.\n\n    min_samples : int > 1 or float between 0 and 1\n        The number of samples in a neighborhood for a point to be considered\n        as a core point. Expressed as an absolute number or a fraction of the\n        number of samples (rounded to be at least 2).\n\n    max_eps : float, default=np.inf\n        The maximum distance between two samples for one to be considered as\n        in the neighborhood of the other. Default value of ``np.inf`` will\n        identify clusters across all scales; reducing ``max_eps`` will result\n        in shorter run times.\n\n    metric : str or callable, default=\'minkowski\'\n        Metric to use for distance computation. Any metric from scikit-learn\n        or scipy.spatial.distance can be used.\n\n        If metric is a callable function, it is called on each\n        pair of instances (rows) and the resulting value recorded. The callable\n        should take two arrays as input and return one value indicating the\n        distance between them. This works for Scipy\'s metrics, but is less\n        efficient than passing the metric name as a string. If metric is\n        "precomputed", X is assumed to be a distance matrix and must be square.\n\n        Valid values for metric are:\n\n        - from scikit-learn: [\'cityblock\', \'cosine\', \'euclidean\', \'l1\', \'l2\',\n          \'manhattan\']\n\n        - from scipy.spatial.distance: [\'braycurtis\', \'canberra\', \'chebyshev\',\n          \'correlation\', \'dice\', \'hamming\', \'jaccard\', \'kulsinski\',\n          \'mahalanobis\', \'minkowski\', \'rogerstanimoto\', \'russellrao\',\n          \'seuclidean\', \'sokalmichener\', \'sokalsneath\', \'sqeuclidean\',\n          \'yule\']\n\n        See the documentation for scipy.spatial.distance for details on these\n        metrics.\n\n        .. note::\n           `\'kulsinski\'` is deprecated from SciPy 1.9 and will be removed in SciPy 1.11.\n\n    p : float, default=2\n        Parameter for the Minkowski metric from\n        :class:`~sklearn.metrics.pairwise_distances`. When p = 1, this is\n        equivalent to using manhattan_distance (l1), and euclidean_distance\n        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.\n\n    metric_params : dict, default=None\n        Additional keyword arguments for the metric function.\n\n    algorithm : {\'auto\', \'ball_tree\', \'kd_tree\', \'brute\'}, default=\'auto\'\n        Algorithm used to compute the nearest neighbors:\n\n        - \'ball_tree\' will use :class:`~sklearn.neighbors.BallTree`.\n        - \'kd_tree\' will use :class:`~sklearn.neighbors.KDTree`.\n        - \'brute\' will use a brute-force search.\n        - \'auto\' will attempt to decide the most appropriate algorithm\n          based on the values passed to `fit` method. (default)\n\n        Note: fitting on sparse input will override the setting of\n        this parameter, using brute force.\n\n    leaf_size : int, default=30\n        Leaf size passed to :class:`~sklearn.neighbors.BallTree` or\n        :class:`~sklearn.neighbors.KDTree`. This can affect the speed of the\n        construction and query, as well as the memory required to store the\n        tree. The optimal value depends on the nature of the problem.\n\n    n_jobs : int, default=None\n        The number of parallel jobs to run for neighbors search.\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    Returns\n    -------\n    ordering_ : array of shape (n_samples,)\n        The cluster ordered list of sample indices.\n\n    core_distances_ : array of shape (n_samples,)\n        Distance at which each sample becomes a core point, indexed by object\n        order. Points which will never be core have a distance of inf. Use\n        ``clust.core_distances_[clust.ordering_]`` to access in cluster order.\n\n    reachability_ : array of shape (n_samples,)\n        Reachability distances per sample, indexed by object order. Use\n        ``clust.reachability_[clust.ordering_]`` to access in cluster order.\n\n    predecessor_ : array of shape (n_samples,)\n        Point that a sample was reached from, indexed by object order.\n        Seed points have a predecessor of -1.\n\n    References\n    ----------\n    .. [1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel,\n       and Jörg Sander. "OPTICS: ordering points to identify the clustering\n       structure." ACM SIGMOD Record 28, no. 2 (1999): 49-60.\n    '
    n_samples = X.shape[0]
    _validate_size(min_samples, n_samples, 'min_samples')
    if min_samples <= 1:
        min_samples = max(2, int(min_samples * n_samples))
    reachability_ = np.empty(n_samples)
    reachability_.fill(np.inf)
    predecessor_ = np.empty(n_samples, dtype=int)
    predecessor_.fill(-1)
    nbrs = NearestNeighbors(n_neighbors=min_samples, algorithm=algorithm, leaf_size=leaf_size, metric=metric, metric_params=metric_params, p=p, n_jobs=n_jobs)
    nbrs.fit(X)
    core_distances_ = _compute_core_distances_(X=X, neighbors=nbrs, min_samples=min_samples, working_memory=None)
    core_distances_[core_distances_ > max_eps] = np.inf
    np.around(core_distances_, decimals=np.finfo(core_distances_.dtype).precision, out=core_distances_)
    processed = np.zeros(X.shape[0], dtype=bool)
    ordering = np.zeros(X.shape[0], dtype=int)
    for ordering_idx in range(X.shape[0]):
        index = np.where(processed == 0)[0]
        point = index[np.argmin(reachability_[index])]
        processed[point] = True
        ordering[ordering_idx] = point
        if core_distances_[point] != np.inf:
            _set_reach_dist(core_distances_=core_distances_, reachability_=reachability_, predecessor_=predecessor_, point_index=point, processed=processed, X=X, nbrs=nbrs, metric=metric, metric_params=metric_params, p=p, max_eps=max_eps)
    if np.all(np.isinf(reachability_)):
        warnings.warn('All reachability values are inf. Set a larger max_eps or all data will be considered outliers.', UserWarning)
    return (ordering, core_distances_, reachability_, predecessor_)

def _set_reach_dist(core_distances_, reachability_, predecessor_, point_index, processed, X, nbrs, metric, metric_params, p, max_eps):
    if False:
        return 10
    P = X[point_index:point_index + 1]
    indices = nbrs.radius_neighbors(P, radius=max_eps, return_distance=False)[0]
    unproc = np.compress(~np.take(processed, indices), indices)
    if not unproc.size:
        return
    if metric == 'precomputed':
        dists = X[[point_index], unproc]
        if isinstance(dists, np.matrix):
            dists = np.asarray(dists)
        dists = dists.ravel()
    else:
        _params = dict() if metric_params is None else metric_params.copy()
        if metric == 'minkowski' and 'p' not in _params:
            _params['p'] = p
        dists = pairwise_distances(P, X[unproc], metric, n_jobs=None, **_params).ravel()
    rdists = np.maximum(dists, core_distances_[point_index])
    np.around(rdists, decimals=np.finfo(rdists.dtype).precision, out=rdists)
    improved = np.where(rdists < np.take(reachability_, unproc))
    reachability_[unproc[improved]] = rdists[improved]
    predecessor_[unproc[improved]] = point_index

@validate_params({'reachability': [np.ndarray], 'core_distances': [np.ndarray], 'ordering': [np.ndarray], 'eps': [Interval(Real, 0, None, closed='both')]}, prefer_skip_nested_validation=True)
def cluster_optics_dbscan(*, reachability, core_distances, ordering, eps):
    if False:
        print('Hello World!')
    'Perform DBSCAN extraction for an arbitrary epsilon.\n\n    Extracting the clusters runs in linear time. Note that this results in\n    ``labels_`` which are close to a :class:`~sklearn.cluster.DBSCAN` with\n    similar settings and ``eps``, only if ``eps`` is close to ``max_eps``.\n\n    Parameters\n    ----------\n    reachability : ndarray of shape (n_samples,)\n        Reachability distances calculated by OPTICS (``reachability_``).\n\n    core_distances : ndarray of shape (n_samples,)\n        Distances at which points become core (``core_distances_``).\n\n    ordering : ndarray of shape (n_samples,)\n        OPTICS ordered point indices (``ordering_``).\n\n    eps : float\n        DBSCAN ``eps`` parameter. Must be set to < ``max_eps``. Results\n        will be close to DBSCAN algorithm if ``eps`` and ``max_eps`` are close\n        to one another.\n\n    Returns\n    -------\n    labels_ : array of shape (n_samples,)\n        The estimated labels.\n    '
    n_samples = len(core_distances)
    labels = np.zeros(n_samples, dtype=int)
    far_reach = reachability > eps
    near_core = core_distances <= eps
    labels[ordering] = np.cumsum(far_reach[ordering] & near_core[ordering]) - 1
    labels[far_reach & ~near_core] = -1
    return labels

@validate_params({'reachability': [np.ndarray], 'predecessor': [np.ndarray], 'ordering': [np.ndarray], 'min_samples': [Interval(Integral, 2, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both')], 'min_cluster_size': [Interval(Integral, 2, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both'), None], 'xi': [Interval(Real, 0, 1, closed='both')], 'predecessor_correction': ['boolean']}, prefer_skip_nested_validation=True)
def cluster_optics_xi(*, reachability, predecessor, ordering, min_samples, min_cluster_size=None, xi=0.05, predecessor_correction=True):
    if False:
        print('Hello World!')
    "Automatically extract clusters according to the Xi-steep method.\n\n    Parameters\n    ----------\n    reachability : ndarray of shape (n_samples,)\n        Reachability distances calculated by OPTICS (`reachability_`).\n\n    predecessor : ndarray of shape (n_samples,)\n        Predecessors calculated by OPTICS.\n\n    ordering : ndarray of shape (n_samples,)\n        OPTICS ordered point indices (`ordering_`).\n\n    min_samples : int > 1 or float between 0 and 1\n        The same as the min_samples given to OPTICS. Up and down steep regions\n        can't have more then ``min_samples`` consecutive non-steep points.\n        Expressed as an absolute number or a fraction of the number of samples\n        (rounded to be at least 2).\n\n    min_cluster_size : int > 1 or float between 0 and 1, default=None\n        Minimum number of samples in an OPTICS cluster, expressed as an\n        absolute number or a fraction of the number of samples (rounded to be\n        at least 2). If ``None``, the value of ``min_samples`` is used instead.\n\n    xi : float between 0 and 1, default=0.05\n        Determines the minimum steepness on the reachability plot that\n        constitutes a cluster boundary. For example, an upwards point in the\n        reachability plot is defined by the ratio from one point to its\n        successor being at most 1-xi.\n\n    predecessor_correction : bool, default=True\n        Correct clusters based on the calculated predecessors.\n\n    Returns\n    -------\n    labels : ndarray of shape (n_samples,)\n        The labels assigned to samples. Points which are not included\n        in any cluster are labeled as -1.\n\n    clusters : ndarray of shape (n_clusters, 2)\n        The list of clusters in the form of ``[start, end]`` in each row, with\n        all indices inclusive. The clusters are ordered according to ``(end,\n        -start)`` (ascending) so that larger clusters encompassing smaller\n        clusters come after such nested smaller clusters. Since ``labels`` does\n        not reflect the hierarchy, usually ``len(clusters) >\n        np.unique(labels)``.\n    "
    n_samples = len(reachability)
    _validate_size(min_samples, n_samples, 'min_samples')
    if min_samples <= 1:
        min_samples = max(2, int(min_samples * n_samples))
    if min_cluster_size is None:
        min_cluster_size = min_samples
    _validate_size(min_cluster_size, n_samples, 'min_cluster_size')
    if min_cluster_size <= 1:
        min_cluster_size = max(2, int(min_cluster_size * n_samples))
    clusters = _xi_cluster(reachability[ordering], predecessor[ordering], ordering, xi, min_samples, min_cluster_size, predecessor_correction)
    labels = _extract_xi_labels(ordering, clusters)
    return (labels, clusters)

def _extend_region(steep_point, xward_point, start, min_samples):
    if False:
        for i in range(10):
            print('nop')
    "Extend the area until it's maximal.\n\n    It's the same function for both upward and downward reagions, depending on\n    the given input parameters. Assuming:\n\n        - steep_{upward/downward}: bool array indicating whether a point is a\n          steep {upward/downward};\n        - upward/downward: bool array indicating whether a point is\n          upward/downward;\n\n    To extend an upward reagion, ``steep_point=steep_upward`` and\n    ``xward_point=downward`` are expected, and to extend a downward region,\n    ``steep_point=steep_downward`` and ``xward_point=upward``.\n\n    Parameters\n    ----------\n    steep_point : ndarray of shape (n_samples,), dtype=bool\n        True if the point is steep downward (upward).\n\n    xward_point : ndarray of shape (n_samples,), dtype=bool\n        True if the point is an upward (respectively downward) point.\n\n    start : int\n        The start of the xward region.\n\n    min_samples : int\n       The same as the min_samples given to OPTICS. Up and down steep\n       regions can't have more then ``min_samples`` consecutive non-steep\n       points.\n\n    Returns\n    -------\n    index : int\n        The current index iterating over all the samples, i.e. where we are up\n        to in our search.\n\n    end : int\n        The end of the region, which can be behind the index. The region\n        includes the ``end`` index.\n    "
    n_samples = len(steep_point)
    non_xward_points = 0
    index = start
    end = start
    while index < n_samples:
        if steep_point[index]:
            non_xward_points = 0
            end = index
        elif not xward_point[index]:
            non_xward_points += 1
            if non_xward_points > min_samples:
                break
        else:
            return end
        index += 1
    return end

def _update_filter_sdas(sdas, mib, xi_complement, reachability_plot):
    if False:
        print('Hello World!')
    'Update steep down areas (SDAs) using the new maximum in between (mib)\n    value, and the given complement of xi, i.e. ``1 - xi``.\n    '
    if np.isinf(mib):
        return []
    res = [sda for sda in sdas if mib <= reachability_plot[sda['start']] * xi_complement]
    for sda in res:
        sda['mib'] = max(sda['mib'], mib)
    return res

def _correct_predecessor(reachability_plot, predecessor_plot, ordering, s, e):
    if False:
        i = 10
        return i + 15
    'Correct for predecessors.\n\n    Applies Algorithm 2 of [1]_.\n\n    Input parameters are ordered by the computer OPTICS ordering.\n\n    .. [1] Schubert, Erich, Michael Gertz.\n       "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of\n       the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.\n    '
    while s < e:
        if reachability_plot[s] > reachability_plot[e]:
            return (s, e)
        p_e = ordering[predecessor_plot[e]]
        for i in range(s, e):
            if p_e == ordering[i]:
                return (s, e)
        e -= 1
    return (None, None)

def _xi_cluster(reachability_plot, predecessor_plot, ordering, xi, min_samples, min_cluster_size, predecessor_correction):
    if False:
        while True:
            i = 10
    "Automatically extract clusters according to the Xi-steep method.\n\n    This is rouphly an implementation of Figure 19 of the OPTICS paper.\n\n    Parameters\n    ----------\n    reachability_plot : array-like of shape (n_samples,)\n        The reachability plot, i.e. reachability ordered according to\n        the calculated ordering, all computed by OPTICS.\n\n    predecessor_plot : array-like of shape (n_samples,)\n        Predecessors ordered according to the calculated ordering.\n\n    xi : float, between 0 and 1\n        Determines the minimum steepness on the reachability plot that\n        constitutes a cluster boundary. For example, an upwards point in the\n        reachability plot is defined by the ratio from one point to its\n        successor being at most 1-xi.\n\n    min_samples : int > 1\n        The same as the min_samples given to OPTICS. Up and down steep regions\n        can't have more then ``min_samples`` consecutive non-steep points.\n\n    min_cluster_size : int > 1\n        Minimum number of samples in an OPTICS cluster.\n\n    predecessor_correction : bool\n        Correct clusters based on the calculated predecessors.\n\n    Returns\n    -------\n    clusters : ndarray of shape (n_clusters, 2)\n        The list of clusters in the form of [start, end] in each row, with all\n        indices inclusive. The clusters are ordered in a way that larger\n        clusters encompassing smaller clusters come after those smaller\n        clusters.\n    "
    reachability_plot = np.hstack((reachability_plot, np.inf))
    xi_complement = 1 - xi
    sdas = []
    clusters = []
    index = 0
    mib = 0.0
    with np.errstate(invalid='ignore'):
        ratio = reachability_plot[:-1] / reachability_plot[1:]
        steep_upward = ratio <= xi_complement
        steep_downward = ratio >= 1 / xi_complement
        downward = ratio > 1
        upward = ratio < 1
    for steep_index in iter(np.flatnonzero(steep_upward | steep_downward)):
        if steep_index < index:
            continue
        mib = max(mib, np.max(reachability_plot[index:steep_index + 1]))
        if steep_downward[steep_index]:
            sdas = _update_filter_sdas(sdas, mib, xi_complement, reachability_plot)
            D_start = steep_index
            D_end = _extend_region(steep_downward, upward, D_start, min_samples)
            D = {'start': D_start, 'end': D_end, 'mib': 0.0}
            sdas.append(D)
            index = D_end + 1
            mib = reachability_plot[index]
        else:
            sdas = _update_filter_sdas(sdas, mib, xi_complement, reachability_plot)
            U_start = steep_index
            U_end = _extend_region(steep_upward, downward, U_start, min_samples)
            index = U_end + 1
            mib = reachability_plot[index]
            U_clusters = []
            for D in sdas:
                c_start = D['start']
                c_end = U_end
                if reachability_plot[c_end + 1] * xi_complement < D['mib']:
                    continue
                D_max = reachability_plot[D['start']]
                if D_max * xi_complement >= reachability_plot[c_end + 1]:
                    while reachability_plot[c_start + 1] > reachability_plot[c_end + 1] and c_start < D['end']:
                        c_start += 1
                elif reachability_plot[c_end + 1] * xi_complement >= D_max:
                    while reachability_plot[c_end - 1] > D_max and c_end > U_start:
                        c_end -= 1
                if predecessor_correction:
                    (c_start, c_end) = _correct_predecessor(reachability_plot, predecessor_plot, ordering, c_start, c_end)
                if c_start is None:
                    continue
                if c_end - c_start + 1 < min_cluster_size:
                    continue
                if c_start > D['end']:
                    continue
                if c_end < U_start:
                    continue
                U_clusters.append((c_start, c_end))
            U_clusters.reverse()
            clusters.extend(U_clusters)
    return np.array(clusters)

def _extract_xi_labels(ordering, clusters):
    if False:
        for i in range(10):
            print('nop')
    'Extracts the labels from the clusters returned by `_xi_cluster`.\n    We rely on the fact that clusters are stored\n    with the smaller clusters coming before the larger ones.\n\n    Parameters\n    ----------\n    ordering : array-like of shape (n_samples,)\n        The ordering of points calculated by OPTICS\n\n    clusters : array-like of shape (n_clusters, 2)\n        List of clusters i.e. (start, end) tuples,\n        as returned by `_xi_cluster`.\n\n    Returns\n    -------\n    labels : ndarray of shape (n_samples,)\n    '
    labels = np.full(len(ordering), -1, dtype=int)
    label = 0
    for c in clusters:
        if not np.any(labels[c[0]:c[1] + 1] != -1):
            labels[c[0]:c[1] + 1] = label
            label += 1
    labels[ordering] = labels.copy()
    return labels