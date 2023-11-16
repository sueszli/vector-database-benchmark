"""
HDBSCAN: Hierarchical Density-Based Spatial Clustering
         of Applications with Noise
"""
from numbers import Integral, Real
from warnings import warn
import numpy as np
from scipy.sparse import csgraph, issparse
from ...base import BaseEstimator, ClusterMixin
from ...metrics import pairwise_distances
from ...metrics._dist_metrics import DistanceMetric
from ...neighbors import BallTree, KDTree, NearestNeighbors
from ...utils._param_validation import Interval, StrOptions
from ...utils.validation import _allclose_dense_sparse, _assert_all_finite
from ._linkage import MST_edge_dtype, make_single_linkage, mst_from_data_matrix, mst_from_mutual_reachability
from ._reachability import mutual_reachability_graph
from ._tree import HIERARCHY_dtype, labelling_at_cut, tree_to_labels
FAST_METRICS = set(KDTree.valid_metrics + BallTree.valid_metrics)
_OUTLIER_ENCODING: dict = {'infinite': {'label': -2, 'prob': 0}, 'missing': {'label': -3, 'prob': np.nan}}

def _brute_mst(mutual_reachability, min_samples):
    if False:
        return 10
    '\n    Builds a minimum spanning tree (MST) from the provided mutual-reachability\n    values. This function dispatches to a custom Cython implementation for\n    dense arrays, and `scipy.sparse.csgraph.minimum_spanning_tree` for sparse\n    arrays/matrices.\n\n    Parameters\n    ----------\n    mututal_reachability_graph: {ndarray, sparse matrix} of shape             (n_samples, n_samples)\n        Weighted adjacency matrix of the mutual reachability graph.\n\n    min_samples : int, default=None\n        The number of samples in a neighborhood for a point\n        to be considered as a core point. This includes the point itself.\n\n    Returns\n    -------\n    mst : ndarray of shape (n_samples - 1,), dtype=MST_edge_dtype\n        The MST representation of the mutual-reachability graph. The MST is\n        represented as a collection of edges.\n    '
    if not issparse(mutual_reachability):
        return mst_from_mutual_reachability(mutual_reachability)
    if csgraph.connected_components(mutual_reachability, directed=False, return_labels=False) > 1:
        raise ValueError(f'There exists points with fewer than {min_samples} neighbors. Ensure your distance matrix has non-zero values for at least `min_sample`={min_samples} neighbors for each points (i.e. K-nn graph), or specify a `max_distance` in `metric_params` to use when distances are missing.')
    sparse_min_spanning_tree = csgraph.minimum_spanning_tree(mutual_reachability)
    (rows, cols) = sparse_min_spanning_tree.nonzero()
    mst = np.core.records.fromarrays([rows, cols, sparse_min_spanning_tree.data], dtype=MST_edge_dtype)
    return mst

def _process_mst(min_spanning_tree):
    if False:
        return 10
    '\n    Builds a single-linkage tree (SLT) from the provided minimum spanning tree\n    (MST). The MST is first sorted then processed by a custom Cython routine.\n\n    Parameters\n    ----------\n    min_spanning_tree : ndarray of shape (n_samples - 1,), dtype=MST_edge_dtype\n        The MST representation of the mutual-reachability graph. The MST is\n        represented as a collection of edges.\n\n    Returns\n    -------\n    single_linkage : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype\n        The single-linkage tree tree (dendrogram) built from the MST.\n    '
    row_order = np.argsort(min_spanning_tree['distance'])
    min_spanning_tree = min_spanning_tree[row_order]
    return make_single_linkage(min_spanning_tree)

def _hdbscan_brute(X, min_samples=5, alpha=None, metric='euclidean', n_jobs=None, copy=False, **metric_params):
    if False:
        print('Hello World!')
    '\n    Builds a single-linkage tree (SLT) from the input data `X`. If\n    `metric="precomputed"` then `X` must be a symmetric array of distances.\n    Otherwise, the pairwise distances are calculated directly and passed to\n    `mutual_reachability_graph`.\n\n    Parameters\n    ----------\n    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)\n        Either the raw data from which to compute the pairwise distances,\n        or the precomputed distances.\n\n    min_samples : int, default=None\n        The number of samples in a neighborhood for a point\n        to be considered as a core point. This includes the point itself.\n\n    alpha : float, default=1.0\n        A distance scaling parameter as used in robust single linkage.\n\n    metric : str or callable, default=\'euclidean\'\n        The metric to use when calculating distance between instances in a\n        feature array.\n\n        - If metric is a string or callable, it must be one of\n          the options allowed by :func:`~sklearn.metrics.pairwise_distances`\n          for its metric parameter.\n\n        - If metric is "precomputed", X is assumed to be a distance matrix and\n          must be square.\n\n    n_jobs : int, default=None\n        The number of jobs to use for computing the pairwise distances. This\n        works by breaking down the pairwise matrix into n_jobs even slices and\n        computing them in parallel. This parameter is passed directly to\n        :func:`~sklearn.metrics.pairwise_distances`.\n\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    copy : bool, default=False\n        If `copy=True` then any time an in-place modifications would be made\n        that would overwrite `X`, a copy will first be made, guaranteeing that\n        the original data will be unchanged. Currently, it only applies when\n        `metric="precomputed"`, when passing a dense array or a CSR sparse\n        array/matrix.\n\n    metric_params : dict, default=None\n        Arguments passed to the distance metric.\n\n    Returns\n    -------\n    single_linkage : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype\n        The single-linkage tree tree (dendrogram) built from the MST.\n    '
    if metric == 'precomputed':
        if X.shape[0] != X.shape[1]:
            raise ValueError(f'The precomputed distance matrix is expected to be symmetric, however it has shape {X.shape}. Please verify that the distance matrix was constructed correctly.')
        if not _allclose_dense_sparse(X, X.T):
            raise ValueError('The precomputed distance matrix is expected to be symmetric, however its values appear to be asymmetric. Please verify that the distance matrix was constructed correctly.')
        distance_matrix = X.copy() if copy else X
    else:
        distance_matrix = pairwise_distances(X, metric=metric, n_jobs=n_jobs, **metric_params)
    distance_matrix /= alpha
    max_distance = metric_params.get('max_distance', 0.0)
    if issparse(distance_matrix) and distance_matrix.format != 'csr':
        distance_matrix = distance_matrix.tocsr()
    mutual_reachability_ = mutual_reachability_graph(distance_matrix, min_samples=min_samples, max_distance=max_distance)
    min_spanning_tree = _brute_mst(mutual_reachability_, min_samples=min_samples)
    if np.isinf(min_spanning_tree['distance']).any():
        warn('The minimum spanning tree contains edge weights with value infinity. Potentially, you are missing too many distances in the initial distance matrix for the given neighborhood size.', UserWarning)
    return _process_mst(min_spanning_tree)

def _hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric='euclidean', leaf_size=40, n_jobs=None, **metric_params):
    if False:
        while True:
            i = 10
    '\n    Builds a single-linkage tree (SLT) from the input data `X`. If\n    `metric="precomputed"` then `X` must be a symmetric array of distances.\n    Otherwise, the pairwise distances are calculated directly and passed to\n    `mutual_reachability_graph`.\n\n    Parameters\n    ----------\n    X : ndarray of shape (n_samples, n_features)\n        The raw data.\n\n    min_samples : int, default=None\n        The number of samples in a neighborhood for a point\n        to be considered as a core point. This includes the point itself.\n\n    alpha : float, default=1.0\n        A distance scaling parameter as used in robust single linkage.\n\n    metric : str or callable, default=\'euclidean\'\n        The metric to use when calculating distance between instances in a\n        feature array. `metric` must be one of the options allowed by\n        :func:`~sklearn.metrics.pairwise_distances` for its metric\n        parameter.\n\n    n_jobs : int, default=None\n        The number of jobs to use for computing the pairwise distances. This\n        works by breaking down the pairwise matrix into n_jobs even slices and\n        computing them in parallel. This parameter is passed directly to\n        :func:`~sklearn.metrics.pairwise_distances`.\n\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    copy : bool, default=False\n        If `copy=True` then any time an in-place modifications would be made\n        that would overwrite `X`, a copy will first be made, guaranteeing that\n        the original data will be unchanged. Currently, it only applies when\n        `metric="precomputed"`, when passing a dense array or a CSR sparse\n        array/matrix.\n\n    metric_params : dict, default=None\n        Arguments passed to the distance metric.\n\n    Returns\n    -------\n    single_linkage : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype\n        The single-linkage tree tree (dendrogram) built from the MST.\n    '
    X = np.asarray(X, order='C')
    nbrs = NearestNeighbors(n_neighbors=min_samples, algorithm=algo, leaf_size=leaf_size, metric=metric, metric_params=metric_params, n_jobs=n_jobs, p=None).fit(X)
    (neighbors_distances, _) = nbrs.kneighbors(X, min_samples, return_distance=True)
    core_distances = np.ascontiguousarray(neighbors_distances[:, -1])
    dist_metric = DistanceMetric.get_metric(metric, **metric_params)
    min_spanning_tree = mst_from_data_matrix(X, core_distances, dist_metric, alpha)
    return _process_mst(min_spanning_tree)

def remap_single_linkage_tree(tree, internal_to_raw, non_finite):
    if False:
        i = 10
        return i + 15
    '\n    Takes an internal single_linkage_tree structure and adds back in a set of points\n    that were initially detected as non-finite and returns that new tree.\n    These points will all be merged into the final node at np.inf distance and\n    considered noise points.\n\n    Parameters\n    ----------\n    tree : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype\n        The single-linkage tree tree (dendrogram) built from the MST.\n    internal_to_raw: dict\n        A mapping from internal integer index to the raw integer index\n    non_finite : ndarray\n        Boolean array of which entries in the raw data are non-finite\n    '
    finite_count = len(internal_to_raw)
    outlier_count = len(non_finite)
    for (i, _) in enumerate(tree):
        left = tree[i]['left_node']
        right = tree[i]['right_node']
        if left < finite_count:
            tree[i]['left_node'] = internal_to_raw[left]
        else:
            tree[i]['left_node'] = left + outlier_count
        if right < finite_count:
            tree[i]['right_node'] = internal_to_raw[right]
        else:
            tree[i]['right_node'] = right + outlier_count
    outlier_tree = np.zeros(len(non_finite), dtype=HIERARCHY_dtype)
    last_cluster_id = max(tree[tree.shape[0] - 1]['left_node'], tree[tree.shape[0] - 1]['right_node'])
    last_cluster_size = tree[tree.shape[0] - 1]['cluster_size']
    for (i, outlier) in enumerate(non_finite):
        outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)
        last_cluster_id += 1
        last_cluster_size += 1
    tree = np.concatenate([tree, outlier_tree])
    return tree

def _get_finite_row_indices(matrix):
    if False:
        return 10
    '\n    Returns the indices of the purely finite rows of a\n    sparse matrix or dense ndarray\n    '
    if issparse(matrix):
        row_indices = np.array([i for (i, row) in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))])
    else:
        (row_indices,) = np.isfinite(matrix.sum(axis=1)).nonzero()
    return row_indices

class HDBSCAN(ClusterMixin, BaseEstimator):
    """Cluster data using hierarchical density-based clustering.

    HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
    with Noise. Performs :class:`~sklearn.cluster.DBSCAN` over varying epsilon
    values and integrates the result to find a clustering that gives the best
    stability over epsilon.
    This allows HDBSCAN to find clusters of varying densities (unlike
    :class:`~sklearn.cluster.DBSCAN`), and be more robust to parameter selection.
    Read more in the :ref:`User Guide <hdbscan>`.

    For an example of how to use HDBSCAN, as well as a comparison to
    :class:`~sklearn.cluster.DBSCAN`, please see the :ref:`plotting demo
    <sphx_glr_auto_examples_cluster_plot_hdbscan.py>`.

    .. versionadded:: 1.3

    Parameters
    ----------
    min_cluster_size : int, default=5
        The minimum number of samples in a group for that group to be
        considered a cluster; groupings smaller than this size will be left
        as noise.

    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        When `None`, defaults to `min_cluster_size`.

    cluster_selection_epsilon : float, default=0.0
        A distance threshold. Clusters below this value will be merged.
        See [5]_ for more information.

    max_cluster_size : int, default=None
        A limit to the size of clusters returned by the `"eom"` cluster
        selection algorithm. There is no limit when `max_cluster_size=None`.
        Has no effect if `cluster_selection_method="leaf"`.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array.

        - If metric is a string or callable, it must be one of
          the options allowed by :func:`~sklearn.metrics.pairwise_distances`
          for its metric parameter.

        - If metric is "precomputed", X is assumed to be a distance matrix and
          must be square.

    metric_params : dict, default=None
        Arguments passed to the distance metric.

    alpha : float, default=1.0
        A distance scaling parameter as used in robust single linkage.
        See [3]_ for more information.

    algorithm : {"auto", "brute", "kd_tree", "ball_tree"}, default="auto"
        Exactly which algorithm to use for computing core distances; By default
        this is set to `"auto"` which attempts to use a
        :class:`~sklearn.neighbors.KDTree` tree if possible, otherwise it uses
        a :class:`~sklearn.neighbors.BallTree` tree. Both `"kd_tree"` and
        `"ball_tree"` algorithms use the
        :class:`~sklearn.neighbors.NearestNeighbors` estimator.

        If the `X` passed during `fit` is sparse or `metric` is invalid for
        both :class:`~sklearn.neighbors.KDTree` and
        :class:`~sklearn.neighbors.BallTree`, then it resolves to use the
        `"brute"` algorithm.

        .. deprecated:: 1.4
           The `'kdtree'` option was deprecated in version 1.4,
           and will be renamed to `'kd_tree'` in 1.6.

        .. deprecated:: 1.4
           The `'balltree'` option was deprecated in version 1.4,
           and will be renamed to `'ball_tree'` in 1.6.

    leaf_size : int, default=40
        Leaf size for trees responsible for fast nearest neighbour queries when
        a KDTree or a BallTree are used as core-distance algorithms. A large
        dataset size and small `leaf_size` may induce excessive memory usage.
        If you are running out of memory consider increasing the `leaf_size`
        parameter. Ignored for `algorithm="brute"`.

    n_jobs : int, default=None
        Number of jobs to run in parallel to calculate distances.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    cluster_selection_method : {"eom", "leaf"}, default="eom"
        The method used to select clusters from the condensed tree. The
        standard approach for HDBSCAN* is to use an Excess of Mass (`"eom"`)
        algorithm to find the most persistent clusters. Alternatively you can
        instead select the clusters at the leaves of the tree -- this provides
        the most fine grained and homogeneous clusters.

    allow_single_cluster : bool, default=False
        By default HDBSCAN* will not produce a single cluster, setting this
        to True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.

    store_centers : str, default=None
        Which, if any, cluster centers to compute and store. The options are:

        - `None` which does not compute nor store any centers.
        - `"centroid"` which calculates the center by taking the weighted
          average of their positions. Note that the algorithm uses the
          euclidean metric and does not guarantee that the output will be
          an observed data point.
        - `"medoid"` which calculates the center by taking the point in the
          fitted data which minimizes the distance to all other points in
          the cluster. This is slower than "centroid" since it requires
          computing additional pairwise distances between points of the
          same cluster but guarantees the output is an observed data point.
          The medoid is also well-defined for arbitrary metrics, and does not
          depend on a euclidean metric.
        - `"both"` which computes and stores both forms of centers.

    copy : bool, default=False
        If `copy=True` then any time an in-place modifications would be made
        that would overwrite data passed to :term:`fit`, a copy will first be
        made, guaranteeing that the original data will be unchanged.
        Currently, it only applies when `metric="precomputed"`, when passing
        a dense array or a CSR sparse matrix and when `algorithm="brute"`.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to :term:`fit`.
        Outliers are labeled as follows:

        - Noisy samples are given the label -1.
        - Samples with infinite elements (+/- np.inf) are given the label -2.
        - Samples with missing data are given the label -3, even if they
          also have infinite elements.

    probabilities_ : ndarray of shape (n_samples,)
        The strength with which each sample is a member of its assigned
        cluster.

        - Clustered samples have probabilities proportional to the degree that
          they persist as part of the cluster.
        - Noisy samples have probability zero.
        - Samples with infinite elements (+/- np.inf) have probability 0.
        - Samples with missing data have probability `np.nan`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    centroids_ : ndarray of shape (n_clusters, n_features)
        A collection containing the centroid of each cluster calculated under
        the standard euclidean metric. The centroids may fall "outside" their
        respective clusters if the clusters themselves are non-convex.

        Note that `n_clusters` only counts non-outlier clusters. That is to
        say, the `-1, -2, -3` labels for the outlier clusters are excluded.

    medoids_ : ndarray of shape (n_clusters, n_features)
        A collection containing the medoid of each cluster calculated under
        the whichever metric was passed to the `metric` parameter. The
        medoids are points in the original cluster which minimize the average
        distance to all other points in that cluster under the chosen metric.
        These can be thought of as the result of projecting the `metric`-based
        centroid back onto the cluster.

        Note that `n_clusters` only counts non-outlier clusters. That is to
        say, the `-1, -2, -3` labels for the outlier clusters are excluded.

    See Also
    --------
    DBSCAN : Density-Based Spatial Clustering of Applications
        with Noise.
    OPTICS : Ordering Points To Identify the Clustering Structure.
    Birch : Memory-efficient, online-learning algorithm.

    References
    ----------

    .. [1] :doi:`Campello, R. J., Moulavi, D., & Sander, J. Density-based clustering
      based on hierarchical density estimates.
      <10.1007/978-3-642-37456-2_14>`
    .. [2] :doi:`Campello, R. J., Moulavi, D., Zimek, A., & Sander, J.
       Hierarchical density estimates for data clustering, visualization,
       and outlier detection.<10.1145/2733381>`

    .. [3] `Chaudhuri, K., & Dasgupta, S. Rates of convergence for the
       cluster tree.
       <https://papers.nips.cc/paper/2010/hash/
       b534ba68236ba543ae44b22bd110a1d6-Abstract.html>`_

    .. [4] `Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and
       Sander, J. Density-Based Clustering Validation.
       <https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf>`_

    .. [5] :arxiv:`Malzer, C., & Baum, M. "A Hybrid Approach To Hierarchical
       Density-based Cluster Selection."<1911.02282>`.

    Examples
    --------
    >>> from sklearn.cluster import HDBSCAN
    >>> from sklearn.datasets import load_digits
    >>> X, _ = load_digits(return_X_y=True)
    >>> hdb = HDBSCAN(min_cluster_size=20)
    >>> hdb.fit(X)
    HDBSCAN(min_cluster_size=20)
    >>> hdb.labels_
    array([ 2,  6, -1, ..., -1, -1, -1])
    """
    _parameter_constraints = {'min_cluster_size': [Interval(Integral, left=2, right=None, closed='left')], 'min_samples': [Interval(Integral, left=1, right=None, closed='left'), None], 'cluster_selection_epsilon': [Interval(Real, left=0, right=None, closed='left')], 'max_cluster_size': [None, Interval(Integral, left=1, right=None, closed='left')], 'metric': [StrOptions(FAST_METRICS | {'precomputed'}), callable], 'metric_params': [dict, None], 'alpha': [Interval(Real, left=0, right=None, closed='neither')], 'algorithm': [StrOptions({'auto', 'brute', 'kd_tree', 'ball_tree', 'kdtree', 'balltree'}, deprecated={'kdtree', 'balltree'})], 'leaf_size': [Interval(Integral, left=1, right=None, closed='left')], 'n_jobs': [Integral, None], 'cluster_selection_method': [StrOptions({'eom', 'leaf'})], 'allow_single_cluster': ['boolean'], 'store_centers': [None, StrOptions({'centroid', 'medoid', 'both'})], 'copy': ['boolean']}

    def __init__(self, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, max_cluster_size=None, metric='euclidean', metric_params=None, alpha=1.0, algorithm='auto', leaf_size=40, n_jobs=None, cluster_selection_method='eom', allow_single_cluster=False, store_centers=None, copy=False):
        if False:
            print('Hello World!')
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.alpha = alpha
        self.max_cluster_size = max_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.store_centers = store_centers
        self.copy = copy

    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        "Find clusters based on hierarchical density-based clustering.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features), or                 ndarray of shape (n_samples, n_samples)\n            A feature array, or array of distances between samples if\n            `metric='precomputed'`.\n\n        y : None\n            Ignored.\n\n        Returns\n        -------\n        self : object\n            Returns self.\n        "
        self._validate_params()
        self._metric_params = self.metric_params or {}
        if self.metric != 'precomputed':
            X = self._validate_data(X, accept_sparse=['csr', 'lil'], force_all_finite=False, dtype=np.float64)
            self._raw_data = X
            all_finite = True
            try:
                _assert_all_finite(X.data if issparse(X) else X)
            except ValueError:
                all_finite = False
            if not all_finite:
                reduced_X = X.sum(axis=1)
                missing_index = np.isnan(reduced_X).nonzero()[0]
                infinite_index = np.isinf(reduced_X).nonzero()[0]
                finite_index = _get_finite_row_indices(X)
                internal_to_raw = {x: y for (x, y) in enumerate(finite_index)}
                X = X[finite_index]
        elif issparse(X):
            X = self._validate_data(X, accept_sparse=['csr', 'lil'], dtype=np.float64)
        else:
            X = self._validate_data(X, force_all_finite=False, dtype=np.float64)
            if np.isnan(X).any():
                raise ValueError('np.nan values found in precomputed-dense')
        if X.shape[0] == 1:
            raise ValueError('n_samples=1 while HDBSCAN requires more than one sample')
        self._min_samples = self.min_cluster_size if self.min_samples is None else self.min_samples
        if self._min_samples > X.shape[0]:
            raise ValueError(f'min_samples ({self._min_samples}) must be at most the number of samples in X ({X.shape[0]})')
        if self.algorithm == 'kdtree':
            warn("`algorithm='kdtree'`has been deprecated in 1.4 and will be renamed to'kd_tree'`in 1.6. To keep the past behaviour, set `algorithm='kd_tree'`.", FutureWarning)
            self.algorithm = 'kd_tree'
        if self.algorithm == 'balltree':
            warn("`algorithm='balltree'`has been deprecated in 1.4 and will be renamed to'ball_tree'`in 1.6. To keep the past behaviour, set `algorithm='ball_tree'`.", FutureWarning)
            self.algorithm = 'ball_tree'
        mst_func = None
        kwargs = dict(X=X, min_samples=self._min_samples, alpha=self.alpha, metric=self.metric, n_jobs=self.n_jobs, **self._metric_params)
        if self.algorithm == 'kd_tree' and self.metric not in KDTree.valid_metrics:
            raise ValueError(f'{self.metric} is not a valid metric for a KDTree-based algorithm. Please select a different metric.')
        elif self.algorithm == 'ball_tree' and self.metric not in BallTree.valid_metrics:
            raise ValueError(f'{self.metric} is not a valid metric for a BallTree-based algorithm. Please select a different metric.')
        if self.algorithm != 'auto':
            if self.metric != 'precomputed' and issparse(X) and (self.algorithm != 'brute'):
                raise ValueError('Sparse data matrices only support algorithm `brute`.')
            if self.algorithm == 'brute':
                mst_func = _hdbscan_brute
                kwargs['copy'] = self.copy
            elif self.algorithm == 'kd_tree':
                mst_func = _hdbscan_prims
                kwargs['algo'] = 'kd_tree'
                kwargs['leaf_size'] = self.leaf_size
            else:
                mst_func = _hdbscan_prims
                kwargs['algo'] = 'ball_tree'
                kwargs['leaf_size'] = self.leaf_size
        elif issparse(X) or self.metric not in FAST_METRICS:
            mst_func = _hdbscan_brute
            kwargs['copy'] = self.copy
        elif self.metric in KDTree.valid_metrics:
            mst_func = _hdbscan_prims
            kwargs['algo'] = 'kd_tree'
            kwargs['leaf_size'] = self.leaf_size
        else:
            mst_func = _hdbscan_prims
            kwargs['algo'] = 'ball_tree'
            kwargs['leaf_size'] = self.leaf_size
        self._single_linkage_tree_ = mst_func(**kwargs)
        (self.labels_, self.probabilities_) = tree_to_labels(self._single_linkage_tree_, self.min_cluster_size, self.cluster_selection_method, self.allow_single_cluster, self.cluster_selection_epsilon, self.max_cluster_size)
        if self.metric != 'precomputed' and (not all_finite):
            self._single_linkage_tree_ = remap_single_linkage_tree(self._single_linkage_tree_, internal_to_raw, non_finite=set(np.hstack([infinite_index, missing_index])))
            new_labels = np.empty(self._raw_data.shape[0], dtype=np.int32)
            new_labels[finite_index] = self.labels_
            new_labels[infinite_index] = _OUTLIER_ENCODING['infinite']['label']
            new_labels[missing_index] = _OUTLIER_ENCODING['missing']['label']
            self.labels_ = new_labels
            new_probabilities = np.zeros(self._raw_data.shape[0], dtype=np.float64)
            new_probabilities[finite_index] = self.probabilities_
            new_probabilities[infinite_index] = _OUTLIER_ENCODING['infinite']['prob']
            new_probabilities[missing_index] = _OUTLIER_ENCODING['missing']['prob']
            self.probabilities_ = new_probabilities
        if self.store_centers:
            self._weighted_cluster_center(X)
        return self

    def fit_predict(self, X, y=None):
        if False:
            print('Hello World!')
        "Cluster X and return the associated cluster labels.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features), or                 ndarray of shape (n_samples, n_samples)\n            A feature array, or array of distances between samples if\n            `metric='precomputed'`.\n\n        y : None\n            Ignored.\n\n        Returns\n        -------\n        y : ndarray of shape (n_samples,)\n            Cluster labels.\n        "
        self.fit(X)
        return self.labels_

    def _weighted_cluster_center(self, X):
        if False:
            return 10
        'Calculate and store the centroids/medoids of each cluster.\n\n        This requires `X` to be a raw feature array, not precomputed\n        distances. Rather than return outputs directly, this helper method\n        instead stores them in the `self.{centroids, medoids}_` attributes.\n        The choice for which attributes are calculated and stored is mediated\n        by the value of `self.store_centers`.\n\n        Parameters\n        ----------\n        X : ndarray of shape (n_samples, n_features)\n            The feature array that the estimator was fit with.\n\n        '
        n_clusters = len(set(self.labels_) - {-1, -2})
        mask = np.empty((X.shape[0],), dtype=np.bool_)
        make_centroids = self.store_centers in ('centroid', 'both')
        make_medoids = self.store_centers in ('medoid', 'both')
        if make_centroids:
            self.centroids_ = np.empty((n_clusters, X.shape[1]), dtype=np.float64)
        if make_medoids:
            self.medoids_ = np.empty((n_clusters, X.shape[1]), dtype=np.float64)
        for idx in range(n_clusters):
            mask = self.labels_ == idx
            data = X[mask]
            strength = self.probabilities_[mask]
            if make_centroids:
                self.centroids_[idx] = np.average(data, weights=strength, axis=0)
            if make_medoids:
                dist_mat = pairwise_distances(data, metric=self.metric, **self._metric_params)
                dist_mat = dist_mat * strength
                medoid_index = np.argmin(dist_mat.sum(axis=1))
                self.medoids_[idx] = data[medoid_index]
        return

    def dbscan_clustering(self, cut_distance, min_cluster_size=5):
        if False:
            return 10
        "Return clustering given by DBSCAN without border points.\n\n        Return clustering that would be equivalent to running DBSCAN* for a\n        particular cut_distance (or epsilon) DBSCAN* can be thought of as\n        DBSCAN without the border points.  As such these results may differ\n        slightly from `cluster.DBSCAN` due to the difference in implementation\n        over the non-core points.\n\n        This can also be thought of as a flat clustering derived from constant\n        height cut through the single linkage tree.\n\n        This represents the result of selecting a cut value for robust single linkage\n        clustering. The `min_cluster_size` allows the flat clustering to declare noise\n        points (and cluster smaller than `min_cluster_size`).\n\n        Parameters\n        ----------\n        cut_distance : float\n            The mutual reachability distance cut value to use to generate a\n            flat clustering.\n\n        min_cluster_size : int, default=5\n            Clusters smaller than this value with be called 'noise' and remain\n            unclustered in the resulting flat clustering.\n\n        Returns\n        -------\n        labels : ndarray of shape (n_samples,)\n            An array of cluster labels, one per datapoint.\n            Outliers are labeled as follows:\n\n            - Noisy samples are given the label -1.\n            - Samples with infinite elements (+/- np.inf) are given the label -2.\n            - Samples with missing data are given the label -3, even if they\n              also have infinite elements.\n        "
        labels = labelling_at_cut(self._single_linkage_tree_, cut_distance, min_cluster_size)
        infinite_index = self.labels_ == _OUTLIER_ENCODING['infinite']['label']
        missing_index = self.labels_ == _OUTLIER_ENCODING['missing']['label']
        labels[infinite_index] = _OUTLIER_ENCODING['infinite']['label']
        labels[missing_index] = _OUTLIER_ENCODING['missing']['label']
        return labels

    def _more_tags(self):
        if False:
            print('Hello World!')
        return {'allow_nan': self.metric != 'precomputed'}