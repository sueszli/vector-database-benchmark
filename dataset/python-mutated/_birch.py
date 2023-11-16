import warnings
from math import sqrt
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from .._config import config_context
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, ClusterMixin, TransformerMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..metrics import pairwise_distances_argmin
from ..metrics.pairwise import euclidean_distances
from ..utils._param_validation import Interval
from ..utils.extmath import row_norms
from ..utils.validation import check_is_fitted
from . import AgglomerativeClustering

def _iterate_sparse_X(X):
    if False:
        print('Hello World!')
    'This little hack returns a densified row when iterating over a sparse\n    matrix, instead of constructing a sparse matrix for every row that is\n    expensive.\n    '
    n_samples = X.shape[0]
    X_indices = X.indices
    X_data = X.data
    X_indptr = X.indptr
    for i in range(n_samples):
        row = np.zeros(X.shape[1])
        (startptr, endptr) = (X_indptr[i], X_indptr[i + 1])
        nonzero_indices = X_indices[startptr:endptr]
        row[nonzero_indices] = X_data[startptr:endptr]
        yield row

def _split_node(node, threshold, branching_factor):
    if False:
        print('Hello World!')
    'The node has to be split if there is no place for a new subcluster\n    in the node.\n    1. Two empty nodes and two empty subclusters are initialized.\n    2. The pair of distant subclusters are found.\n    3. The properties of the empty subclusters and nodes are updated\n       according to the nearest distance between the subclusters to the\n       pair of distant subclusters.\n    4. The two nodes are set as children to the two subclusters.\n    '
    new_subcluster1 = _CFSubcluster()
    new_subcluster2 = _CFSubcluster()
    new_node1 = _CFNode(threshold=threshold, branching_factor=branching_factor, is_leaf=node.is_leaf, n_features=node.n_features, dtype=node.init_centroids_.dtype)
    new_node2 = _CFNode(threshold=threshold, branching_factor=branching_factor, is_leaf=node.is_leaf, n_features=node.n_features, dtype=node.init_centroids_.dtype)
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2
    if node.is_leaf:
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2
    dist = euclidean_distances(node.centroids_, Y_norm_squared=node.squared_norm_, squared=True)
    n_clusters = dist.shape[0]
    farthest_idx = np.unravel_index(dist.argmax(), (n_clusters, n_clusters))
    (node1_dist, node2_dist) = dist[farthest_idx,]
    node1_closer = node1_dist < node2_dist
    node1_closer[farthest_idx[0]] = True
    for (idx, subcluster) in enumerate(node.subclusters_):
        if node1_closer[idx]:
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
    return (new_subcluster1, new_subcluster2)

class _CFNode:
    """Each node in a CFTree is called a CFNode.

    The CFNode can have a maximum of branching_factor
    number of CFSubclusters.

    Parameters
    ----------
    threshold : float
        Threshold needed for a new subcluster to enter a CFSubcluster.

    branching_factor : int
        Maximum number of CF subclusters in each node.

    is_leaf : bool
        We need to know if the CFNode is a leaf or not, in order to
        retrieve the final subclusters.

    n_features : int
        The number of features.

    Attributes
    ----------
    subclusters_ : list
        List of subclusters for a particular CFNode.

    prev_leaf_ : _CFNode
        Useful only if is_leaf is True.

    next_leaf_ : _CFNode
        next_leaf. Useful only if is_leaf is True.
        the final subclusters.

    init_centroids_ : ndarray of shape (branching_factor + 1, n_features)
        Manipulate ``init_centroids_`` throughout rather than centroids_ since
        the centroids are just a view of the ``init_centroids_`` .

    init_sq_norm_ : ndarray of shape (branching_factor + 1,)
        manipulate init_sq_norm_ throughout. similar to ``init_centroids_``.

    centroids_ : ndarray of shape (branching_factor + 1, n_features)
        View of ``init_centroids_``.

    squared_norm_ : ndarray of shape (branching_factor + 1,)
        View of ``init_sq_norm_``.

    """

    def __init__(self, *, threshold, branching_factor, is_leaf, n_features, dtype):
        if False:
            print('Hello World!')
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.n_features = n_features
        self.subclusters_ = []
        self.init_centroids_ = np.zeros((branching_factor + 1, n_features), dtype=dtype)
        self.init_sq_norm_ = np.zeros(branching_factor + 1, dtype)
        self.squared_norm_ = []
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def append_subcluster(self, subcluster):
        if False:
            return 10
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        self.init_centroids_[n_samples] = subcluster.centroid_
        self.init_sq_norm_[n_samples] = subcluster.sq_norm_
        self.centroids_ = self.init_centroids_[:n_samples + 1, :]
        self.squared_norm_ = self.init_sq_norm_[:n_samples + 1]

    def update_split_subclusters(self, subcluster, new_subcluster1, new_subcluster2):
        if False:
            while True:
                i = 10
        'Remove a subcluster from a node and update it with the\n        split subclusters.\n        '
        ind = self.subclusters_.index(subcluster)
        self.subclusters_[ind] = new_subcluster1
        self.init_centroids_[ind] = new_subcluster1.centroid_
        self.init_sq_norm_[ind] = new_subcluster1.sq_norm_
        self.append_subcluster(new_subcluster2)

    def insert_cf_subcluster(self, subcluster):
        if False:
            while True:
                i = 10
        'Insert a new subcluster into the node.'
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False
        threshold = self.threshold
        branching_factor = self.branching_factor
        dist_matrix = np.dot(self.centroids_, subcluster.centroid_)
        dist_matrix *= -2.0
        dist_matrix += self.squared_norm_
        closest_index = np.argmin(dist_matrix)
        closest_subcluster = self.subclusters_[closest_index]
        if closest_subcluster.child_ is not None:
            split_child = closest_subcluster.child_.insert_cf_subcluster(subcluster)
            if not split_child:
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = self.subclusters_[closest_index].centroid_
                self.init_sq_norm_[closest_index] = self.subclusters_[closest_index].sq_norm_
                return False
            else:
                (new_subcluster1, new_subcluster2) = _split_node(closest_subcluster.child_, threshold, branching_factor)
                self.update_split_subclusters(closest_subcluster, new_subcluster1, new_subcluster2)
                if len(self.subclusters_) > self.branching_factor:
                    return True
                return False
        else:
            merged = closest_subcluster.merge_subcluster(subcluster, self.threshold)
            if merged:
                self.init_centroids_[closest_index] = closest_subcluster.centroid_
                self.init_sq_norm_[closest_index] = closest_subcluster.sq_norm_
                return False
            elif len(self.subclusters_) < self.branching_factor:
                self.append_subcluster(subcluster)
                return False
            else:
                self.append_subcluster(subcluster)
                return True

class _CFSubcluster:
    """Each subcluster in a CFNode is called a CFSubcluster.

    A CFSubcluster can have a CFNode has its child.

    Parameters
    ----------
    linear_sum : ndarray of shape (n_features,), default=None
        Sample. This is kept optional to allow initialization of empty
        subclusters.

    Attributes
    ----------
    n_samples_ : int
        Number of samples that belong to each subcluster.

    linear_sum_ : ndarray
        Linear sum of all the samples in a subcluster. Prevents holding
        all sample data in memory.

    squared_sum_ : float
        Sum of the squared l2 norms of all samples belonging to a subcluster.

    centroid_ : ndarray of shape (branching_factor + 1, n_features)
        Centroid of the subcluster. Prevent recomputing of centroids when
        ``CFNode.centroids_`` is called.

    child_ : _CFNode
        Child Node of the subcluster. Once a given _CFNode is set as the child
        of the _CFNode, it is set to ``self.child_``.

    sq_norm_ : ndarray of shape (branching_factor + 1,)
        Squared norm of the subcluster. Used to prevent recomputing when
        pairwise minimum distances are computed.
    """

    def __init__(self, *, linear_sum=None):
        if False:
            print('Hello World!')
        if linear_sum is None:
            self.n_samples_ = 0
            self.squared_sum_ = 0.0
            self.centroid_ = self.linear_sum_ = 0
        else:
            self.n_samples_ = 1
            self.centroid_ = self.linear_sum_ = linear_sum
            self.squared_sum_ = self.sq_norm_ = np.dot(self.linear_sum_, self.linear_sum_)
        self.child_ = None

    def update(self, subcluster):
        if False:
            print('Hello World!')
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum_ += subcluster.linear_sum_
        self.squared_sum_ += subcluster.squared_sum_
        self.centroid_ = self.linear_sum_ / self.n_samples_
        self.sq_norm_ = np.dot(self.centroid_, self.centroid_)

    def merge_subcluster(self, nominee_cluster, threshold):
        if False:
            i = 10
            return i + 15
        'Check if a cluster is worthy enough to be merged. If\n        yes then merge.\n        '
        new_ss = self.squared_sum_ + nominee_cluster.squared_sum_
        new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
        new_n = self.n_samples_ + nominee_cluster.n_samples_
        new_centroid = 1 / new_n * new_ls
        new_sq_norm = np.dot(new_centroid, new_centroid)
        sq_radius = new_ss / new_n - new_sq_norm
        if sq_radius <= threshold ** 2:
            (self.n_samples_, self.linear_sum_, self.squared_sum_, self.centroid_, self.sq_norm_) = (new_n, new_ls, new_ss, new_centroid, new_sq_norm)
            return True
        return False

    @property
    def radius(self):
        if False:
            for i in range(10):
                print('nop')
        'Return radius of the subcluster'
        sq_radius = self.squared_sum_ / self.n_samples_ - self.sq_norm_
        return sqrt(max(0, sq_radius))

class Birch(ClassNamePrefixFeaturesOutMixin, ClusterMixin, TransformerMixin, BaseEstimator):
    """Implements the BIRCH clustering algorithm.

    It is a memory-efficient, online-learning algorithm provided as an
    alternative to :class:`MiniBatchKMeans`. It constructs a tree
    data structure with the cluster centroids being read off the leaf.
    These can be either the final cluster centroids or can be provided as input
    to another clustering algorithm such as :class:`AgglomerativeClustering`.

    Read more in the :ref:`User Guide <birch>`.

    .. versionadded:: 0.16

    Parameters
    ----------
    threshold : float, default=0.5
        The radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be lesser than the threshold. Otherwise a new
        subcluster is started. Setting this value to be very low promotes
        splitting and vice-versa.

    branching_factor : int, default=50
        Maximum number of CF subclusters in each node. If a new samples enters
        such that the number of subclusters exceed the branching_factor then
        that node is split into two nodes with the subclusters redistributed
        in each. The parent subcluster of that node is removed and two new
        subclusters are added as parents of the 2 split nodes.

    n_clusters : int, instance of sklearn.cluster model or None, default=3
        Number of clusters after the final clustering step, which treats the
        subclusters from the leaves as new samples.

        - `None` : the final clustering step is not performed and the
          subclusters are returned as they are.

        - :mod:`sklearn.cluster` Estimator : If a model is provided, the model
          is fit treating the subclusters as new samples and the initial data
          is mapped to the label of the closest subcluster.

        - `int` : the model fit is :class:`AgglomerativeClustering` with
          `n_clusters` set to be equal to the int.

    compute_labels : bool, default=True
        Whether or not to compute labels for each fit.

    copy : bool, default=True
        Whether or not to make a copy of the given data. If set to False,
        the initial data will be overwritten.

    Attributes
    ----------
    root_ : _CFNode
        Root of the CFTree.

    dummy_leaf_ : _CFNode
        Start pointer to all the leaves.

    subcluster_centers_ : ndarray
        Centroids of all subclusters read directly from the leaves.

    subcluster_labels_ : ndarray
        Labels assigned to the centroids of the subclusters after
        they are clustered globally.

    labels_ : ndarray of shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MiniBatchKMeans : Alternative implementation that does incremental updates
        of the centers' positions using mini-batches.

    Notes
    -----
    The tree data structure consists of nodes with each node consisting of
    a number of subclusters. The maximum number of subclusters in a node
    is determined by the branching factor. Each subcluster maintains a
    linear sum, squared sum and the number of samples in that subcluster.
    In addition, each subcluster can also have a node as its child, if the
    subcluster is not a member of a leaf node.

    For a new point entering the root, it is merged with the subcluster closest
    to it and the linear sum, squared sum and the number of samples of that
    subcluster are updated. This is done recursively till the properties of
    the leaf node are updated.

    References
    ----------
    * Tian Zhang, Raghu Ramakrishnan, Maron Livny
      BIRCH: An efficient data clustering method for large databases.
      https://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

    * Roberto Perdisci
      JBirch - Java implementation of BIRCH clustering algorithm
      https://code.google.com/archive/p/jbirch

    Examples
    --------
    >>> from sklearn.cluster import Birch
    >>> X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
    >>> brc = Birch(n_clusters=None)
    >>> brc.fit(X)
    Birch(n_clusters=None)
    >>> brc.predict(X)
    array([0, 0, 0, 1, 1, 1])
    """
    _parameter_constraints: dict = {'threshold': [Interval(Real, 0.0, None, closed='neither')], 'branching_factor': [Interval(Integral, 1, None, closed='neither')], 'n_clusters': [None, ClusterMixin, Interval(Integral, 1, None, closed='left')], 'compute_labels': ['boolean'], 'copy': ['boolean']}

    def __init__(self, *, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True):
        if False:
            for i in range(10):
                print('nop')
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.copy = copy

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        '\n        Build a CF Tree for the input data.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Input data.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        Returns\n        -------\n        self\n            Fitted estimator.\n        '
        return self._fit(X, partial=False)

    def _fit(self, X, partial):
        if False:
            while True:
                i = 10
        has_root = getattr(self, 'root_', None)
        first_call = not (partial and has_root)
        X = self._validate_data(X, accept_sparse='csr', copy=self.copy, reset=first_call, dtype=[np.float64, np.float32])
        threshold = self.threshold
        branching_factor = self.branching_factor
        (n_samples, n_features) = X.shape
        if first_call:
            self.root_ = _CFNode(threshold=threshold, branching_factor=branching_factor, is_leaf=True, n_features=n_features, dtype=X.dtype)
            self.dummy_leaf_ = _CFNode(threshold=threshold, branching_factor=branching_factor, is_leaf=True, n_features=n_features, dtype=X.dtype)
            self.dummy_leaf_.next_leaf_ = self.root_
            self.root_.prev_leaf_ = self.dummy_leaf_
        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X
        for sample in iter_func(X):
            subcluster = _CFSubcluster(linear_sum=sample)
            split = self.root_.insert_cf_subcluster(subcluster)
            if split:
                (new_subcluster1, new_subcluster2) = _split_node(self.root_, threshold, branching_factor)
                del self.root_
                self.root_ = _CFNode(threshold=threshold, branching_factor=branching_factor, is_leaf=False, n_features=n_features, dtype=X.dtype)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)
        centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()])
        self.subcluster_centers_ = centroids
        self._n_features_out = self.subcluster_centers_.shape[0]
        self._global_clustering(X)
        return self

    def _get_leaves(self):
        if False:
            print('Hello World!')
        '\n        Retrieve the leaves of the CF Node.\n\n        Returns\n        -------\n        leaves : list of shape (n_leaves,)\n            List of the leaf nodes.\n        '
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X=None, y=None):
        if False:
            i = 10
            return i + 15
        '\n        Online learning. Prevents rebuilding of CFTree from scratch.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features),             default=None\n            Input data. If X is not provided, only the global clustering\n            step is done.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        Returns\n        -------\n        self\n            Fitted estimator.\n        '
        if X is None:
            self._global_clustering()
            return self
        else:
            return self._fit(X, partial=True)

    def _check_fit(self, X):
        if False:
            i = 10
            return i + 15
        check_is_fitted(self)
        if hasattr(self, 'subcluster_centers_') and X.shape[1] != self.subcluster_centers_.shape[1]:
            raise ValueError('Training data and predicted data do not have same number of features.')

    def predict(self, X):
        if False:
            return 10
        '\n        Predict data using the ``centroids_`` of subclusters.\n\n        Avoid computation of the row norms of X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Input data.\n\n        Returns\n        -------\n        labels : ndarray of shape(n_samples,)\n            Labelled data.\n        '
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse='csr', reset=False)
        return self._predict(X)

    def _predict(self, X):
        if False:
            print('Hello World!')
        'Predict data using the ``centroids_`` of subclusters.'
        kwargs = {'Y_norm_squared': self._subcluster_norms}
        with config_context(assume_finite=True):
            argmin = pairwise_distances_argmin(X, self.subcluster_centers_, metric_kwargs=kwargs)
        return self.subcluster_labels_[argmin]

    def transform(self, X):
        if False:
            i = 10
            return i + 15
        '\n        Transform X into subcluster centroids dimension.\n\n        Each dimension represents the distance from the sample point to each\n        cluster centroid.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Input data.\n\n        Returns\n        -------\n        X_trans : {array-like, sparse matrix} of shape (n_samples, n_clusters)\n            Transformed data.\n        '
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse='csr', reset=False)
        with config_context(assume_finite=True):
            return euclidean_distances(X, self.subcluster_centers_)

    def _global_clustering(self, X=None):
        if False:
            i = 10
            return i + 15
        '\n        Global clustering for the subclusters obtained after fitting\n        '
        clusterer = self.n_clusters
        centroids = self.subcluster_centers_
        compute_labels = X is not None and self.compute_labels
        not_enough_centroids = False
        if isinstance(clusterer, Integral):
            clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
            if len(centroids) < self.n_clusters:
                not_enough_centroids = True
        self._subcluster_norms = row_norms(self.subcluster_centers_, squared=True)
        if clusterer is None or not_enough_centroids:
            self.subcluster_labels_ = np.arange(len(centroids))
            if not_enough_centroids:
                warnings.warn('Number of subclusters found (%d) by BIRCH is less than (%d). Decrease the threshold.' % (len(centroids), self.n_clusters), ConvergenceWarning)
        else:
            self.subcluster_labels_ = clusterer.fit_predict(self.subcluster_centers_)
        if compute_labels:
            self.labels_ = self._predict(X)

    def _more_tags(self):
        if False:
            while True:
                i = 10
        return {'preserves_dtype': [np.float64, np.float32]}