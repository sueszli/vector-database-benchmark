"""K-means clustering."""
import warnings
from abc import ABC, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, ClusterMixin, TransformerMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import _euclidean_distances, euclidean_distances
from ..utils import check_array, check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, stable_cumsum
from ..utils.fixes import threadpool_info, threadpool_limits
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.sparsefuncs_fast import assign_rows_csr
from ..utils.validation import _check_sample_weight, _is_arraylike_not_scalar, check_is_fitted
from ._k_means_common import CHUNK_SIZE, _inertia_dense, _inertia_sparse, _is_same_clustering
from ._k_means_elkan import elkan_iter_chunked_dense, elkan_iter_chunked_sparse, init_bounds_dense, init_bounds_sparse
from ._k_means_lloyd import lloyd_iter_chunked_dense, lloyd_iter_chunked_sparse
from ._k_means_minibatch import _minibatch_update_dense, _minibatch_update_sparse

@validate_params({'X': ['array-like', 'sparse matrix'], 'n_clusters': [Interval(Integral, 1, None, closed='left')], 'sample_weight': ['array-like', None], 'x_squared_norms': ['array-like', None], 'random_state': ['random_state'], 'n_local_trials': [Interval(Integral, 1, None, closed='left'), None]}, prefer_skip_nested_validation=True)
def kmeans_plusplus(X, n_clusters, *, sample_weight=None, x_squared_norms=None, random_state=None, n_local_trials=None):
    if False:
        print('Hello World!')
    'Init n_clusters seeds according to k-means++.\n\n    .. versionadded:: 0.24\n\n    Parameters\n    ----------\n    X : {array-like, sparse matrix} of shape (n_samples, n_features)\n        The data to pick seeds from.\n\n    n_clusters : int\n        The number of centroids to initialize.\n\n    sample_weight : array-like of shape (n_samples,), default=None\n        The weights for each observation in `X`. If `None`, all observations\n        are assigned equal weight. `sample_weight` is ignored if `init`\n        is a callable or a user provided array.\n\n        .. versionadded:: 1.3\n\n    x_squared_norms : array-like of shape (n_samples,), default=None\n        Squared Euclidean norm of each data point.\n\n    random_state : int or RandomState instance, default=None\n        Determines random number generation for centroid initialization. Pass\n        an int for reproducible output across multiple function calls.\n        See :term:`Glossary <random_state>`.\n\n    n_local_trials : int, default=None\n        The number of seeding trials for each center (except the first),\n        of which the one reducing inertia the most is greedily chosen.\n        Set to None to make the number of trials depend logarithmically\n        on the number of seeds (2+log(k)) which is the recommended setting.\n        Setting to 1 disables the greedy cluster selection and recovers the\n        vanilla k-means++ algorithm which was empirically shown to work less\n        well than its greedy variant.\n\n    Returns\n    -------\n    centers : ndarray of shape (n_clusters, n_features)\n        The initial centers for k-means.\n\n    indices : ndarray of shape (n_clusters,)\n        The index location of the chosen centers in the data array X. For a\n        given index and center, X[index] = center.\n\n    Notes\n    -----\n    Selects initial cluster centers for k-mean clustering in a smart way\n    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.\n    "k-means++: the advantages of careful seeding". ACM-SIAM symposium\n    on Discrete algorithms. 2007\n\n    Examples\n    --------\n\n    >>> from sklearn.cluster import kmeans_plusplus\n    >>> import numpy as np\n    >>> X = np.array([[1, 2], [1, 4], [1, 0],\n    ...               [10, 2], [10, 4], [10, 0]])\n    >>> centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=0)\n    >>> centers\n    array([[10,  2],\n           [ 1,  0]])\n    >>> indices\n    array([3, 2])\n    '
    check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
    if X.shape[0] < n_clusters:
        raise ValueError(f'n_samples={X.shape[0]} should be >= n_clusters={n_clusters}.')
    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)
    else:
        x_squared_norms = check_array(x_squared_norms, dtype=X.dtype, ensure_2d=False)
    if x_squared_norms.shape[0] != X.shape[0]:
        raise ValueError(f'The length of x_squared_norms {x_squared_norms.shape[0]} should be equal to the length of n_samples {X.shape[0]}.')
    random_state = check_random_state(random_state)
    (centers, indices) = _kmeans_plusplus(X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials)
    return (centers, indices)

def _kmeans_plusplus(X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials=None):
    if False:
        return 10
    'Computational component for initialization of n_clusters by\n    k-means++. Prior validation of data is assumed.\n\n    Parameters\n    ----------\n    X : {ndarray, sparse matrix} of shape (n_samples, n_features)\n        The data to pick seeds for.\n\n    n_clusters : int\n        The number of seeds to choose.\n\n    sample_weight : ndarray of shape (n_samples,)\n        The weights for each observation in `X`.\n\n    x_squared_norms : ndarray of shape (n_samples,)\n        Squared Euclidean norm of each data point.\n\n    random_state : RandomState instance\n        The generator used to initialize the centers.\n        See :term:`Glossary <random_state>`.\n\n    n_local_trials : int, default=None\n        The number of seeding trials for each center (except the first),\n        of which the one reducing inertia the most is greedily chosen.\n        Set to None to make the number of trials depend logarithmically\n        on the number of seeds (2+log(k)); this is the default.\n\n    Returns\n    -------\n    centers : ndarray of shape (n_clusters, n_features)\n        The initial centers for k-means.\n\n    indices : ndarray of shape (n_clusters,)\n        The index location of the chosen centers in the data array X. For a\n        given index and center, X[index] = center.\n    '
    (n_samples, n_features) = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))
    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    indices = np.full(n_clusters, -1, dtype=int)
    if sp.issparse(X):
        centers[0] = X[[center_id]].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id
    closest_dist_sq = _euclidean_distances(centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True)
    current_pot = closest_dist_sq @ sample_weight
    for c in range(1, n_clusters):
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(sample_weight * closest_dist_sq), rand_vals)
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)
        distance_to_candidates = _euclidean_distances(X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]
        if sp.issparse(X):
            centers[c] = X[[best_candidate]].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate
    return (centers, indices)

def _tolerance(X, tol):
    if False:
        for i in range(10):
            print('nop')
    'Return a tolerance which is dependent on the dataset.'
    if tol == 0:
        return 0
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol

@validate_params({'X': ['array-like', 'sparse matrix'], 'sample_weight': ['array-like', None], 'return_n_iter': [bool]}, prefer_skip_nested_validation=False)
def k_means(X, n_clusters, *, sample_weight=None, init='k-means++', n_init='warn', max_iter=300, verbose=False, tol=0.0001, random_state=None, copy_x=True, algorithm='lloyd', return_n_iter=False):
    if False:
        return 10
    'Perform K-means clustering algorithm.\n\n    Read more in the :ref:`User Guide <k_means>`.\n\n    Parameters\n    ----------\n    X : {array-like, sparse matrix} of shape (n_samples, n_features)\n        The observations to cluster. It must be noted that the data\n        will be converted to C ordering, which will cause a memory copy\n        if the given data is not C-contiguous.\n\n    n_clusters : int\n        The number of clusters to form as well as the number of\n        centroids to generate.\n\n    sample_weight : array-like of shape (n_samples,), default=None\n        The weights for each observation in `X`. If `None`, all observations\n        are assigned equal weight. `sample_weight` is not used during\n        initialization if `init` is a callable or a user provided array.\n\n    init : {\'k-means++\', \'random\'}, callable or array-like of shape             (n_clusters, n_features), default=\'k-means++\'\n        Method for initialization:\n\n        - `\'k-means++\'` : selects initial cluster centers for k-mean\n          clustering in a smart way to speed up convergence. See section\n          Notes in k_init for more details.\n        - `\'random\'`: choose `n_clusters` observations (rows) at random from data\n          for the initial centroids.\n        - If an array is passed, it should be of shape `(n_clusters, n_features)`\n          and gives the initial centers.\n        - If a callable is passed, it should take arguments `X`, `n_clusters` and a\n          random state and return an initialization.\n\n    n_init : \'auto\' or int, default=10\n        Number of time the k-means algorithm will be run with different\n        centroid seeds. The final results will be the best output of\n        n_init consecutive runs in terms of inertia.\n\n        When `n_init=\'auto\'`, the number of runs depends on the value of init:\n        10 if using `init=\'random\'` or `init` is a callable;\n        1 if using `init=\'k-means++\'` or `init` is an array-like.\n\n        .. versionadded:: 1.2\n           Added \'auto\' option for `n_init`.\n\n        .. versionchanged:: 1.4\n           Default value for `n_init` will change from 10 to `\'auto\'` in version 1.4.\n\n    max_iter : int, default=300\n        Maximum number of iterations of the k-means algorithm to run.\n\n    verbose : bool, default=False\n        Verbosity mode.\n\n    tol : float, default=1e-4\n        Relative tolerance with regards to Frobenius norm of the difference\n        in the cluster centers of two consecutive iterations to declare\n        convergence.\n\n    random_state : int, RandomState instance or None, default=None\n        Determines random number generation for centroid initialization. Use\n        an int to make the randomness deterministic.\n        See :term:`Glossary <random_state>`.\n\n    copy_x : bool, default=True\n        When pre-computing distances it is more numerically accurate to center\n        the data first. If `copy_x` is True (default), then the original data is\n        not modified. If False, the original data is modified, and put back\n        before the function returns, but small numerical differences may be\n        introduced by subtracting and then adding the data mean. Note that if\n        the original data is not C-contiguous, a copy will be made even if\n        `copy_x` is False. If the original data is sparse, but not in CSR format,\n        a copy will be made even if `copy_x` is False.\n\n    algorithm : {"lloyd", "elkan", "auto", "full"}, default="lloyd"\n        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.\n        The `"elkan"` variation can be more efficient on some datasets with\n        well-defined clusters, by using the triangle inequality. However it\'s\n        more memory intensive due to the allocation of an extra array of shape\n        `(n_samples, n_clusters)`.\n\n        `"auto"` and `"full"` are deprecated and they will be removed in\n        Scikit-Learn 1.3. They are both aliases for `"lloyd"`.\n\n        .. versionchanged:: 0.18\n            Added Elkan algorithm\n\n        .. versionchanged:: 1.1\n            Renamed "full" to "lloyd", and deprecated "auto" and "full".\n            Changed "auto" to use "lloyd" instead of "elkan".\n\n    return_n_iter : bool, default=False\n        Whether or not to return the number of iterations.\n\n    Returns\n    -------\n    centroid : ndarray of shape (n_clusters, n_features)\n        Centroids found at the last iteration of k-means.\n\n    label : ndarray of shape (n_samples,)\n        The `label[i]` is the code or index of the centroid the\n        i\'th observation is closest to.\n\n    inertia : float\n        The final value of the inertia criterion (sum of squared distances to\n        the closest centroid for all observations in the training set).\n\n    best_n_iter : int\n        Number of iterations corresponding to the best results.\n        Returned only if `return_n_iter` is set to True.\n    '
    est = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, verbose=verbose, tol=tol, random_state=random_state, copy_x=copy_x, algorithm=algorithm).fit(X, sample_weight=sample_weight)
    if return_n_iter:
        return (est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_)
    else:
        return (est.cluster_centers_, est.labels_, est.inertia_)

def _kmeans_single_elkan(X, sample_weight, centers_init, max_iter=300, verbose=False, tol=0.0001, n_threads=1):
    if False:
        while True:
            i = 10
    "A single run of k-means elkan, assumes preparation completed prior.\n\n    Parameters\n    ----------\n    X : {ndarray, sparse matrix} of shape (n_samples, n_features)\n        The observations to cluster. If sparse matrix, must be in CSR format.\n\n    sample_weight : array-like of shape (n_samples,)\n        The weights for each observation in X.\n\n    centers_init : ndarray of shape (n_clusters, n_features)\n        The initial centers.\n\n    max_iter : int, default=300\n        Maximum number of iterations of the k-means algorithm to run.\n\n    verbose : bool, default=False\n        Verbosity mode.\n\n    tol : float, default=1e-4\n        Relative tolerance with regards to Frobenius norm of the difference\n        in the cluster centers of two consecutive iterations to declare\n        convergence.\n        It's not advised to set `tol=0` since convergence might never be\n        declared due to rounding errors. Use a very small number instead.\n\n    n_threads : int, default=1\n        The number of OpenMP threads to use for the computation. Parallelism is\n        sample-wise on the main cython loop which assigns each sample to its\n        closest center.\n\n    Returns\n    -------\n    centroid : ndarray of shape (n_clusters, n_features)\n        Centroids found at the last iteration of k-means.\n\n    label : ndarray of shape (n_samples,)\n        label[i] is the code or index of the centroid the\n        i'th observation is closest to.\n\n    inertia : float\n        The final value of the inertia criterion (sum of squared distances to\n        the closest centroid for all observations in the training set).\n\n    n_iter : int\n        Number of iterations run.\n    "
    n_samples = X.shape[0]
    n_clusters = centers_init.shape[0]
    centers = centers_init
    centers_new = np.zeros_like(centers)
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()
    center_half_distances = euclidean_distances(centers) / 2
    distance_next_center = np.partition(np.asarray(center_half_distances), kth=1, axis=0)[1]
    upper_bounds = np.zeros(n_samples, dtype=X.dtype)
    lower_bounds = np.zeros((n_samples, n_clusters), dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)
    if sp.issparse(X):
        init_bounds = init_bounds_sparse
        elkan_iter = elkan_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        init_bounds = init_bounds_dense
        elkan_iter = elkan_iter_chunked_dense
        _inertia = _inertia_dense
    init_bounds(X, centers, center_half_distances, labels, upper_bounds, lower_bounds, n_threads=n_threads)
    strict_convergence = False
    for i in range(max_iter):
        elkan_iter(X, sample_weight, centers, centers_new, weight_in_clusters, center_half_distances, distance_next_center, upper_bounds, lower_bounds, labels, center_shift, n_threads)
        center_half_distances = euclidean_distances(centers_new) / 2
        distance_next_center = np.partition(np.asarray(center_half_distances), kth=1, axis=0)[1]
        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels, n_threads)
            print(f'Iteration {i}, inertia {inertia}')
        (centers, centers_new) = (centers_new, centers)
        if np.array_equal(labels, labels_old):
            if verbose:
                print(f'Converged at iteration {i}: strict convergence.')
            strict_convergence = True
            break
        else:
            center_shift_tot = (center_shift ** 2).sum()
            if center_shift_tot <= tol:
                if verbose:
                    print(f'Converged at iteration {i}: center shift {center_shift_tot} within tolerance {tol}.')
                break
        labels_old[:] = labels
    if not strict_convergence:
        elkan_iter(X, sample_weight, centers, centers, weight_in_clusters, center_half_distances, distance_next_center, upper_bounds, lower_bounds, labels, center_shift, n_threads, update_centers=False)
    inertia = _inertia(X, sample_weight, centers, labels, n_threads)
    return (labels, inertia, centers, i + 1)

def _kmeans_single_lloyd(X, sample_weight, centers_init, max_iter=300, verbose=False, tol=0.0001, n_threads=1):
    if False:
        print('Hello World!')
    "A single run of k-means lloyd, assumes preparation completed prior.\n\n    Parameters\n    ----------\n    X : {ndarray, sparse matrix} of shape (n_samples, n_features)\n        The observations to cluster. If sparse matrix, must be in CSR format.\n\n    sample_weight : ndarray of shape (n_samples,)\n        The weights for each observation in X.\n\n    centers_init : ndarray of shape (n_clusters, n_features)\n        The initial centers.\n\n    max_iter : int, default=300\n        Maximum number of iterations of the k-means algorithm to run.\n\n    verbose : bool, default=False\n        Verbosity mode\n\n    tol : float, default=1e-4\n        Relative tolerance with regards to Frobenius norm of the difference\n        in the cluster centers of two consecutive iterations to declare\n        convergence.\n        It's not advised to set `tol=0` since convergence might never be\n        declared due to rounding errors. Use a very small number instead.\n\n    n_threads : int, default=1\n        The number of OpenMP threads to use for the computation. Parallelism is\n        sample-wise on the main cython loop which assigns each sample to its\n        closest center.\n\n    Returns\n    -------\n    centroid : ndarray of shape (n_clusters, n_features)\n        Centroids found at the last iteration of k-means.\n\n    label : ndarray of shape (n_samples,)\n        label[i] is the code or index of the centroid the\n        i'th observation is closest to.\n\n    inertia : float\n        The final value of the inertia criterion (sum of squared distances to\n        the closest centroid for all observations in the training set).\n\n    n_iter : int\n        Number of iterations run.\n    "
    n_clusters = centers_init.shape[0]
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)
    if sp.issparse(X):
        lloyd_iter = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        lloyd_iter = lloyd_iter_chunked_dense
        _inertia = _inertia_dense
    strict_convergence = False
    with threadpool_limits(limits=1, user_api='blas'):
        for i in range(max_iter):
            lloyd_iter(X, sample_weight, centers, centers_new, weight_in_clusters, labels, center_shift, n_threads)
            if verbose:
                inertia = _inertia(X, sample_weight, centers, labels, n_threads)
                print(f'Iteration {i}, inertia {inertia}.')
            (centers, centers_new) = (centers_new, centers)
            if np.array_equal(labels, labels_old):
                if verbose:
                    print(f'Converged at iteration {i}: strict convergence.')
                strict_convergence = True
                break
            else:
                center_shift_tot = (center_shift ** 2).sum()
                if center_shift_tot <= tol:
                    if verbose:
                        print(f'Converged at iteration {i}: center shift {center_shift_tot} within tolerance {tol}.')
                    break
            labels_old[:] = labels
        if not strict_convergence:
            lloyd_iter(X, sample_weight, centers, centers, weight_in_clusters, labels, center_shift, n_threads, update_centers=False)
    inertia = _inertia(X, sample_weight, centers, labels, n_threads)
    return (labels, inertia, centers, i + 1)

def _labels_inertia(X, sample_weight, centers, n_threads=1, return_inertia=True):
    if False:
        while True:
            i = 10
    'E step of the K-means EM algorithm.\n\n    Compute the labels and the inertia of the given samples and centers.\n\n    Parameters\n    ----------\n    X : {ndarray, sparse matrix} of shape (n_samples, n_features)\n        The input samples to assign to the labels. If sparse matrix, must\n        be in CSR format.\n\n    sample_weight : ndarray of shape (n_samples,)\n        The weights for each observation in X.\n\n    x_squared_norms : ndarray of shape (n_samples,)\n        Precomputed squared euclidean norm of each data point, to speed up\n        computations.\n\n    centers : ndarray of shape (n_clusters, n_features)\n        The cluster centers.\n\n    n_threads : int, default=1\n        The number of OpenMP threads to use for the computation. Parallelism is\n        sample-wise on the main cython loop which assigns each sample to its\n        closest center.\n\n    return_inertia : bool, default=True\n        Whether to compute and return the inertia.\n\n    Returns\n    -------\n    labels : ndarray of shape (n_samples,)\n        The resulting assignment.\n\n    inertia : float\n        Sum of squared distances of samples to their closest cluster center.\n        Inertia is only returned if return_inertia is True.\n    '
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]
    labels = np.full(n_samples, -1, dtype=np.int32)
    center_shift = np.zeros(n_clusters, dtype=centers.dtype)
    if sp.issparse(X):
        _labels = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        _labels = lloyd_iter_chunked_dense
        _inertia = _inertia_dense
    _labels(X, sample_weight, centers, centers_new=None, weight_in_clusters=None, labels=labels, center_shift=center_shift, n_threads=n_threads, update_centers=False)
    if return_inertia:
        inertia = _inertia(X, sample_weight, centers, labels, n_threads)
        return (labels, inertia)
    return labels

def _labels_inertia_threadpool_limit(X, sample_weight, centers, n_threads=1, return_inertia=True):
    if False:
        for i in range(10):
            print('nop')
    'Same as _labels_inertia but in a threadpool_limits context.'
    with threadpool_limits(limits=1, user_api='blas'):
        result = _labels_inertia(X, sample_weight, centers, n_threads, return_inertia)
    return result

class _BaseKMeans(ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC):
    """Base class for KMeans and MiniBatchKMeans"""
    _parameter_constraints: dict = {'n_clusters': [Interval(Integral, 1, None, closed='left')], 'init': [StrOptions({'k-means++', 'random'}), callable, 'array-like'], 'n_init': [StrOptions({'auto'}), Hidden(StrOptions({'warn'})), Interval(Integral, 1, None, closed='left')], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'tol': [Interval(Real, 0, None, closed='left')], 'verbose': ['verbose'], 'random_state': ['random_state']}

    def __init__(self, n_clusters, *, init, n_init, max_iter, tol, verbose, random_state):
        if False:
            return 10
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state

    def _check_params_vs_input(self, X, default_n_init=None):
        if False:
            i = 10
            return i + 15
        if X.shape[0] < self.n_clusters:
            raise ValueError(f'n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}.')
        self._tol = _tolerance(X, self.tol)
        self._n_init = self.n_init
        if self._n_init == 'warn':
            warnings.warn(f"The default value of `n_init` will change from {default_n_init} to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning", FutureWarning, stacklevel=2)
            self._n_init = default_n_init
        if self._n_init == 'auto':
            if isinstance(self.init, str) and self.init == 'k-means++':
                self._n_init = 1
            elif isinstance(self.init, str) and self.init == 'random':
                self._n_init = default_n_init
            elif callable(self.init):
                self._n_init = default_n_init
            else:
                self._n_init = 1
        if _is_arraylike_not_scalar(self.init) and self._n_init != 1:
            warnings.warn(f'Explicit initial center position passed: performing only one init in {self.__class__.__name__} instead of n_init={self._n_init}.', RuntimeWarning, stacklevel=2)
            self._n_init = 1

    @abstractmethod
    def _warn_mkl_vcomp(self, n_active_threads):
        if False:
            return 10
        'Issue an estimator specific warning when vcomp and mkl are both present\n\n        This method is called by `_check_mkl_vcomp`.\n        '

    def _check_mkl_vcomp(self, X, n_samples):
        if False:
            return 10
        'Check when vcomp and mkl are both present'
        if sp.issparse(X):
            return
        n_active_threads = int(np.ceil(n_samples / CHUNK_SIZE))
        if n_active_threads < self._n_threads:
            modules = threadpool_info()
            has_vcomp = 'vcomp' in [module['prefix'] for module in modules]
            has_mkl = ('mkl', 'intel') in [(module['internal_api'], module.get('threading_layer', None)) for module in modules]
            if has_vcomp and has_mkl:
                self._warn_mkl_vcomp(n_active_threads)

    def _validate_center_shape(self, X, centers):
        if False:
            while True:
                i = 10
        'Check if centers is compatible with X and n_clusters.'
        if centers.shape[0] != self.n_clusters:
            raise ValueError(f'The shape of the initial centers {centers.shape} does not match the number of clusters {self.n_clusters}.')
        if centers.shape[1] != X.shape[1]:
            raise ValueError(f'The shape of the initial centers {centers.shape} does not match the number of features of the data {X.shape[1]}.')

    def _check_test_data(self, X):
        if False:
            for i in range(10):
                print('nop')
        X = self._validate_data(X, accept_sparse='csr', reset=False, dtype=[np.float64, np.float32], order='C', accept_large_sparse=False)
        return X

    def _init_centroids(self, X, x_squared_norms, init, random_state, sample_weight, init_size=None, n_centroids=None):
        if False:
            print('Hello World!')
        "Compute the initial centroids.\n\n        Parameters\n        ----------\n        X : {ndarray, sparse matrix} of shape (n_samples, n_features)\n            The input samples.\n\n        x_squared_norms : ndarray of shape (n_samples,)\n            Squared euclidean norm of each data point. Pass it if you have it\n            at hands already to avoid it being recomputed here.\n\n        init : {'k-means++', 'random'}, callable or ndarray of shape                 (n_clusters, n_features)\n            Method for initialization.\n\n        random_state : RandomState instance\n            Determines random number generation for centroid initialization.\n            See :term:`Glossary <random_state>`.\n\n        sample_weight : ndarray of shape (n_samples,)\n            The weights for each observation in X. `sample_weight` is not used\n            during initialization if `init` is a callable or a user provided\n            array.\n\n        init_size : int, default=None\n            Number of samples to randomly sample for speeding up the\n            initialization (sometimes at the expense of accuracy).\n\n        n_centroids : int, default=None\n            Number of centroids to initialize.\n            If left to 'None' the number of centroids will be equal to\n            number of clusters to form (self.n_clusters).\n\n        Returns\n        -------\n        centers : ndarray of shape (n_clusters, n_features)\n            Initial centroids of clusters.\n        "
        n_samples = X.shape[0]
        n_clusters = self.n_clusters if n_centroids is None else n_centroids
        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]
            sample_weight = sample_weight[init_indices]
        if isinstance(init, str) and init == 'k-means++':
            (centers, _) = _kmeans_plusplus(X, n_clusters, random_state=random_state, x_squared_norms=x_squared_norms, sample_weight=sample_weight)
        elif isinstance(init, str) and init == 'random':
            seeds = random_state.choice(n_samples, size=n_clusters, replace=False, p=sample_weight / sample_weight.sum())
            centers = X[seeds]
        elif _is_arraylike_not_scalar(self.init):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(centers, dtype=X.dtype, copy=False, order='C')
            self._validate_center_shape(X, centers)
        if sp.issparse(centers):
            centers = centers.toarray()
        return centers

    def fit_predict(self, X, y=None, sample_weight=None):
        if False:
            for i in range(10):
                print('nop')
        'Compute cluster centers and predict cluster index for each sample.\n\n        Convenience method; equivalent to calling fit(X) followed by\n        predict(X).\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            New data to transform.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            The weights for each observation in X. If None, all observations\n            are assigned equal weight.\n\n        Returns\n        -------\n        labels : ndarray of shape (n_samples,)\n            Index of the cluster each sample belongs to.\n        '
        return self.fit(X, sample_weight=sample_weight).labels_

    def predict(self, X, sample_weight='deprecated'):
        if False:
            i = 10
            return i + 15
        'Predict the closest cluster each sample in X belongs to.\n\n        In the vector quantization literature, `cluster_centers_` is called\n        the code book and each value returned by `predict` is the index of\n        the closest code in the code book.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            New data to predict.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            The weights for each observation in X. If None, all observations\n            are assigned equal weight.\n\n            .. deprecated:: 1.3\n               The parameter `sample_weight` is deprecated in version 1.3\n               and will be removed in 1.5.\n\n        Returns\n        -------\n        labels : ndarray of shape (n_samples,)\n            Index of the cluster each sample belongs to.\n        '
        check_is_fitted(self)
        X = self._check_test_data(X)
        if not (isinstance(sample_weight, str) and sample_weight == 'deprecated'):
            warnings.warn("'sample_weight' was deprecated in version 1.3 and will be removed in 1.5.", FutureWarning)
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        else:
            sample_weight = _check_sample_weight(None, X, dtype=X.dtype)
        labels = _labels_inertia_threadpool_limit(X, sample_weight, self.cluster_centers_, n_threads=self._n_threads, return_inertia=False)
        return labels

    def fit_transform(self, X, y=None, sample_weight=None):
        if False:
            while True:
                i = 10
        'Compute clustering and transform X to cluster-distance space.\n\n        Equivalent to fit(X).transform(X), but more efficiently implemented.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            New data to transform.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            The weights for each observation in X. If None, all observations\n            are assigned equal weight.\n\n        Returns\n        -------\n        X_new : ndarray of shape (n_samples, n_clusters)\n            X transformed in the new space.\n        '
        return self.fit(X, sample_weight=sample_weight)._transform(X)

    def transform(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Transform X to a cluster-distance space.\n\n        In the new space, each dimension is the distance to the cluster\n        centers. Note that even if X is sparse, the array returned by\n        `transform` will typically be dense.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            New data to transform.\n\n        Returns\n        -------\n        X_new : ndarray of shape (n_samples, n_clusters)\n            X transformed in the new space.\n        '
        check_is_fitted(self)
        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        if False:
            i = 10
            return i + 15
        'Guts of transform method; no input validation.'
        return euclidean_distances(X, self.cluster_centers_)

    def score(self, X, y=None, sample_weight=None):
        if False:
            return 10
        'Opposite of the value of X on the K-means objective.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            New data.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            The weights for each observation in X. If None, all observations\n            are assigned equal weight.\n\n        Returns\n        -------\n        score : float\n            Opposite of the value of X on the K-means objective.\n        '
        check_is_fitted(self)
        X = self._check_test_data(X)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        (_, scores) = _labels_inertia_threadpool_limit(X, sample_weight, self.cluster_centers_, self._n_threads)
        return -scores

    def _more_tags(self):
        if False:
            print('Hello World!')
        return {'_xfail_checks': {'check_sample_weights_invariance': 'zero sample_weight is not equivalent to removing samples'}}

class KMeans(_BaseKMeans):
    """K-Means clustering.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape             (n_clusters, n_features), default='k-means++'
        Method for initialization:

        * 'k-means++' : selects initial cluster centroids using sampling             based on an empirical probability distribution of the points'             contribution to the overall inertia. This technique speeds up             convergence. The algorithm implemented is "greedy k-means++". It             differs from the vanilla k-means++ by making several trials at             each sampling step and choosing the best centroid among them.

        * 'random': choose `n_clusters` observations (rows) at random from         data for the initial centroids.

        * If an array is passed, it should be of shape (n_clusters, n_features)        and gives the initial centers.

        * If a callable is passed, it should take arguments X, n_clusters and a        random state and return an initialization.

        For an example of how to use the different `init` strategy, see the example
        entitled :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_digits.py`.

    n_init : 'auto' or int, default=10
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` will change from 10 to `'auto'` in version 1.4.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan", "auto", "full"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

        `"auto"` and `"full"` are deprecated and they will be removed in
        Scikit-Learn 1.3. They are both aliases for `"lloyd"`.

        .. versionchanged:: 0.18
            Added Elkan algorithm

        .. versionchanged:: 1.1
            Renamed "full" to "lloyd", and deprecated "auto" and "full".
            Changed "auto" to use "lloyd" instead of "elkan".

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations run.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MiniBatchKMeans : Alternative online implementation that does incremental
        updates of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.

    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.

    The average complexity is given by O(k n T), where n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features.
    Refer to :doi:`"How slow is the k-means method?" D. Arthur and S. Vassilvitskii -
    SoCG2006.<10.1145/1137856.1137880>` for more details.

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.

    Examples
    --------

    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    """
    _parameter_constraints: dict = {**_BaseKMeans._parameter_constraints, 'copy_x': ['boolean'], 'algorithm': [StrOptions({'lloyd', 'elkan', 'auto', 'full'}, deprecated={'auto', 'full'})]}

    def __init__(self, n_clusters=8, *, init='k-means++', n_init='warn', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
        if False:
            print('Hello World!')
        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state)
        self.copy_x = copy_x
        self.algorithm = algorithm

    def _check_params_vs_input(self, X):
        if False:
            for i in range(10):
                print('nop')
        super()._check_params_vs_input(X, default_n_init=10)
        self._algorithm = self.algorithm
        if self._algorithm in ('auto', 'full'):
            warnings.warn(f"algorithm='{self._algorithm}' is deprecated, it will be removed in 1.3. Using 'lloyd' instead.", FutureWarning)
            self._algorithm = 'lloyd'
        if self._algorithm == 'elkan' and self.n_clusters == 1:
            warnings.warn("algorithm='elkan' doesn't make sense for a single cluster. Using 'lloyd' instead.", RuntimeWarning)
            self._algorithm = 'lloyd'

    def _warn_mkl_vcomp(self, n_active_threads):
        if False:
            print('Hello World!')
        'Warn when vcomp and mkl are both present'
        warnings.warn(f'KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS={n_active_threads}.')

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        if False:
            for i in range(10):
                print('nop')
        "Compute k-means clustering.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training instances to cluster. It must be noted that the data\n            will be converted to C ordering, which will cause a memory\n            copy if the given data is not C-contiguous.\n            If a sparse matrix is passed, a copy will be made if it's not in\n            CSR format.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            The weights for each observation in X. If None, all observations\n            are assigned equal weight. `sample_weight` is not used during\n            initialization if `init` is a callable or a user provided array.\n\n            .. versionadded:: 0.20\n\n        Returns\n        -------\n        self : object\n            Fitted estimator.\n        "
        X = self._validate_data(X, accept_sparse='csr', dtype=[np.float64, np.float32], order='C', copy=self.copy_x, accept_large_sparse=False)
        self._check_params_vs_input(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()
        init = self.init
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like:
            init = check_array(init, dtype=X.dtype, copy=True, order='C')
            self._validate_center_shape(X, init)
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            X -= X_mean
            if init_is_array_like:
                init -= X_mean
        x_squared_norms = row_norms(X, squared=True)
        if self._algorithm == 'elkan':
            kmeans_single = _kmeans_single_elkan
        else:
            kmeans_single = _kmeans_single_lloyd
            self._check_mkl_vcomp(X, X.shape[0])
        (best_inertia, best_labels) = (None, None)
        for i in range(self._n_init):
            centers_init = self._init_centroids(X, x_squared_norms=x_squared_norms, init=init, random_state=random_state, sample_weight=sample_weight)
            if self.verbose:
                print('Initialization complete')
            (labels, inertia, centers, n_iter_) = kmeans_single(X, sample_weight, centers_init, max_iter=self.max_iter, verbose=self.verbose, tol=self._tol, n_threads=self._n_threads)
            if best_inertia is None or (inertia < best_inertia and (not _is_same_clustering(labels, best_labels, self.n_clusters))):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_
        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean
        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn('Number of distinct clusters ({}) found smaller than n_clusters ({}). Possibly due to duplicate points in X.'.format(distinct_clusters, self.n_clusters), ConvergenceWarning, stacklevel=2)
        self.cluster_centers_ = best_centers
        self._n_features_out = self.cluster_centers_.shape[0]
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

def _mini_batch_step(X, sample_weight, centers, centers_new, weight_sums, random_state, random_reassign=False, reassignment_ratio=0.01, verbose=False, n_threads=1):
    if False:
        i = 10
        return i + 15
    'Incremental update of the centers for the Minibatch K-Means algorithm.\n\n    Parameters\n    ----------\n\n    X : {ndarray, sparse matrix} of shape (n_samples, n_features)\n        The original data array. If sparse, must be in CSR format.\n\n    x_squared_norms : ndarray of shape (n_samples,)\n        Squared euclidean norm of each data point.\n\n    sample_weight : ndarray of shape (n_samples,)\n        The weights for each observation in `X`.\n\n    centers : ndarray of shape (n_clusters, n_features)\n        The cluster centers before the current iteration\n\n    centers_new : ndarray of shape (n_clusters, n_features)\n        The cluster centers after the current iteration. Modified in-place.\n\n    weight_sums : ndarray of shape (n_clusters,)\n        The vector in which we keep track of the numbers of points in a\n        cluster. This array is modified in place.\n\n    random_state : RandomState instance\n        Determines random number generation for low count centers reassignment.\n        See :term:`Glossary <random_state>`.\n\n    random_reassign : boolean, default=False\n        If True, centers with very low counts are randomly reassigned\n        to observations.\n\n    reassignment_ratio : float, default=0.01\n        Control the fraction of the maximum number of counts for a\n        center to be reassigned. A higher value means that low count\n        centers are more likely to be reassigned, which means that the\n        model will take longer to converge, but should converge in a\n        better clustering.\n\n    verbose : bool, default=False\n        Controls the verbosity.\n\n    n_threads : int, default=1\n        The number of OpenMP threads to use for the computation.\n\n    Returns\n    -------\n    inertia : float\n        Sum of squared distances of samples to their closest cluster center.\n        The inertia is computed after finding the labels and before updating\n        the centers.\n    '
    (labels, inertia) = _labels_inertia(X, sample_weight, centers, n_threads=n_threads)
    if sp.issparse(X):
        _minibatch_update_sparse(X, sample_weight, centers, centers_new, weight_sums, labels, n_threads)
    else:
        _minibatch_update_dense(X, sample_weight, centers, centers_new, weight_sums, labels, n_threads)
    if random_reassign and reassignment_ratio > 0:
        to_reassign = weight_sums < reassignment_ratio * weight_sums.max()
        if to_reassign.sum() > 0.5 * X.shape[0]:
            indices_dont_reassign = np.argsort(weight_sums)[int(0.5 * X.shape[0]):]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()
        if n_reassigns:
            new_centers = random_state.choice(X.shape[0], replace=False, size=n_reassigns)
            if verbose:
                print(f'[MiniBatchKMeans] Reassigning {n_reassigns} cluster centers.')
            if sp.issparse(X):
                assign_rows_csr(X, new_centers.astype(np.intp, copy=False), np.where(to_reassign)[0].astype(np.intp, copy=False), centers_new)
            else:
                centers_new[to_reassign] = X[new_centers]
        weight_sums[to_reassign] = np.min(weight_sums[~to_reassign])
    return inertia

class MiniBatchKMeans(_BaseKMeans):
    """
    Mini-Batch K-Means clustering.

    Read more in the :ref:`User Guide <mini_batch_kmeans>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape             (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

    batch_size : int, default=1024
        Size of the mini batches.
        For faster computations, you can set the ``batch_size`` greater than
        256 * number of cores to enable parallelism on all cores.

        .. versionchanged:: 1.0
           `batch_size` default changed from 100 to 1024.

    verbose : int, default=0
        Verbosity mode.

    compute_labels : bool, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.

        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.

        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.

        If `None`, the heuristic is `init_size = 3 * batch_size` if
        `3 * batch_size < n_clusters`, else `init_size = 3 * n_clusters`.

    n_init : 'auto' or int, default=3
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the best of
        the `n_init` initializations as measured by inertia. Several runs are
        recommended for sparse high-dimensional problems (see
        :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        3 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` will change from 3 to `'auto'` in version 1.4.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a center to
        be reassigned. A higher value means that low count centers are more
        easily reassigned, which means that the model will take longer to
        converge, but should converge in a better clustering. However, too high
        a value may cause convergence issues, especially with a small batch
        size.

    Attributes
    ----------

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point (if compute_labels is set to True).

    inertia_ : float
        The value of the inertia criterion associated with the chosen
        partition if compute_labels is set to True. If compute_labels is set to
        False, it's an approximation of the inertia based on an exponentially
        weighted average of the batch inertiae.
        The inertia is defined as the sum of square distances of samples to
        their cluster center, weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations over the full dataset.

    n_steps_ : int
        Number of minibatches processed.

        .. versionadded:: 1.0

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    KMeans : The classic implementation of the clustering method based on the
        Lloyd's algorithm. It consumes the whole set of input data at each
        iteration.

    Notes
    -----
    See https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

    When there are too few points in the dataset, some centers may be
    duplicated, which means that a proper clustering in terms of the number
    of requesting clusters and the number of returned clusters will not
    always match. One solution is to set `reassignment_ratio=0`, which
    prevents reassignments of clusters that are too small.

    Examples
    --------
    >>> from sklearn.cluster import MiniBatchKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 0], [4, 4],
    ...               [4, 5], [0, 1], [2, 2],
    ...               [3, 2], [5, 5], [1, -1]])
    >>> # manually fit on batches
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          n_init="auto")
    >>> kmeans = kmeans.partial_fit(X[0:6,:])
    >>> kmeans = kmeans.partial_fit(X[6:12,:])
    >>> kmeans.cluster_centers_
    array([[3.375, 3.  ],
           [0.75 , 0.5 ]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([1, 0], dtype=int32)
    >>> # fit on the whole data
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          max_iter=10,
    ...                          n_init="auto").fit(X)
    >>> kmeans.cluster_centers_
    array([[3.55102041, 2.48979592],
           [1.06896552, 1.        ]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([1, 0], dtype=int32)
    """
    _parameter_constraints: dict = {**_BaseKMeans._parameter_constraints, 'batch_size': [Interval(Integral, 1, None, closed='left')], 'compute_labels': ['boolean'], 'max_no_improvement': [Interval(Integral, 0, None, closed='left'), None], 'init_size': [Interval(Integral, 1, None, closed='left'), None], 'reassignment_ratio': [Interval(Real, 0, None, closed='left')]}

    def __init__(self, n_clusters=8, *, init='k-means++', max_iter=100, batch_size=1024, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init='warn', reassignment_ratio=0.01):
        if False:
            return 10
        super().__init__(n_clusters=n_clusters, init=init, max_iter=max_iter, verbose=verbose, random_state=random_state, tol=tol, n_init=n_init)
        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.compute_labels = compute_labels
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio

    def _check_params_vs_input(self, X):
        if False:
            print('Hello World!')
        super()._check_params_vs_input(X, default_n_init=3)
        self._batch_size = min(self.batch_size, X.shape[0])
        self._init_size = self.init_size
        if self._init_size is None:
            self._init_size = 3 * self._batch_size
            if self._init_size < self.n_clusters:
                self._init_size = 3 * self.n_clusters
        elif self._init_size < self.n_clusters:
            warnings.warn(f'init_size={self._init_size} should be larger than n_clusters={self.n_clusters}. Setting it to min(3*n_clusters, n_samples)', RuntimeWarning, stacklevel=2)
            self._init_size = 3 * self.n_clusters
        self._init_size = min(self._init_size, X.shape[0])
        if self.reassignment_ratio < 0:
            raise ValueError(f'reassignment_ratio should be >= 0, got {self.reassignment_ratio} instead.')

    def _warn_mkl_vcomp(self, n_active_threads):
        if False:
            print('Hello World!')
        'Warn when vcomp and mkl are both present'
        warnings.warn(f'MiniBatchKMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can prevent it by setting batch_size >= {self._n_threads * CHUNK_SIZE} or by setting the environment variable OMP_NUM_THREADS={n_active_threads}')

    def _mini_batch_convergence(self, step, n_steps, n_samples, centers_squared_diff, batch_inertia):
        if False:
            return 10
        'Helper function to encapsulate the early stopping logic'
        batch_inertia /= self._batch_size
        step = step + 1
        if step == 1:
            if self.verbose:
                print(f'Minibatch step {step}/{n_steps}: mean batch inertia: {batch_inertia}')
            return False
        if self._ewa_inertia is None:
            self._ewa_inertia = batch_inertia
        else:
            alpha = self._batch_size * 2.0 / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_inertia = self._ewa_inertia * (1 - alpha) + batch_inertia * alpha
        if self.verbose:
            print(f'Minibatch step {step}/{n_steps}: mean batch inertia: {batch_inertia}, ewa inertia: {self._ewa_inertia}')
        if self._tol > 0.0 and centers_squared_diff <= self._tol:
            if self.verbose:
                print(f'Converged (small centers change) at step {step}/{n_steps}')
            return True
        if self._ewa_inertia_min is None or self._ewa_inertia < self._ewa_inertia_min:
            self._no_improvement = 0
            self._ewa_inertia_min = self._ewa_inertia
        else:
            self._no_improvement += 1
        if self.max_no_improvement is not None and self._no_improvement >= self.max_no_improvement:
            if self.verbose:
                print(f'Converged (lack of improvement in inertia) at step {step}/{n_steps}')
            return True
        return False

    def _random_reassign(self):
        if False:
            i = 10
            return i + 15
        'Check if a random reassignment needs to be done.\n\n        Do random reassignments each time 10 * n_clusters samples have been\n        processed.\n\n        If there are empty clusters we always want to reassign.\n        '
        self._n_since_last_reassign += self._batch_size
        if (self._counts == 0).any() or self._n_since_last_reassign >= 10 * self.n_clusters:
            self._n_since_last_reassign = 0
            return True
        return False

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        if False:
            while True:
                i = 10
        "Compute the centroids on X by chunking it into mini-batches.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training instances to cluster. It must be noted that the data\n            will be converted to C ordering, which will cause a memory copy\n            if the given data is not C-contiguous.\n            If a sparse matrix is passed, a copy will be made if it's not in\n            CSR format.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            The weights for each observation in X. If None, all observations\n            are assigned equal weight. `sample_weight` is not used during\n            initialization if `init` is a callable or a user provided array.\n\n            .. versionadded:: 0.20\n\n        Returns\n        -------\n        self : object\n            Fitted estimator.\n        "
        X = self._validate_data(X, accept_sparse='csr', dtype=[np.float64, np.float32], order='C', accept_large_sparse=False)
        self._check_params_vs_input(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()
        (n_samples, n_features) = X.shape
        init = self.init
        if _is_arraylike_not_scalar(init):
            init = check_array(init, dtype=X.dtype, copy=True, order='C')
            self._validate_center_shape(X, init)
        self._check_mkl_vcomp(X, self._batch_size)
        x_squared_norms = row_norms(X, squared=True)
        validation_indices = random_state.randint(0, n_samples, self._init_size)
        X_valid = X[validation_indices]
        sample_weight_valid = sample_weight[validation_indices]
        best_inertia = None
        for init_idx in range(self._n_init):
            if self.verbose:
                print(f'Init {init_idx + 1}/{self._n_init} with method {init}')
            cluster_centers = self._init_centroids(X, x_squared_norms=x_squared_norms, init=init, random_state=random_state, init_size=self._init_size, sample_weight=sample_weight)
            (_, inertia) = _labels_inertia_threadpool_limit(X_valid, sample_weight_valid, cluster_centers, n_threads=self._n_threads)
            if self.verbose:
                print(f'Inertia for init {init_idx + 1}/{self._n_init}: {inertia}')
            if best_inertia is None or inertia < best_inertia:
                init_centers = cluster_centers
                best_inertia = inertia
        centers = init_centers
        centers_new = np.empty_like(centers)
        self._counts = np.zeros(self.n_clusters, dtype=X.dtype)
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0
        self._n_since_last_reassign = 0
        n_steps = self.max_iter * n_samples // self._batch_size
        with threadpool_limits(limits=1, user_api='blas'):
            for i in range(n_steps):
                minibatch_indices = random_state.randint(0, n_samples, self._batch_size)
                batch_inertia = _mini_batch_step(X=X[minibatch_indices], sample_weight=sample_weight[minibatch_indices], centers=centers, centers_new=centers_new, weight_sums=self._counts, random_state=random_state, random_reassign=self._random_reassign(), reassignment_ratio=self.reassignment_ratio, verbose=self.verbose, n_threads=self._n_threads)
                if self._tol > 0.0:
                    centers_squared_diff = np.sum((centers_new - centers) ** 2)
                else:
                    centers_squared_diff = 0
                (centers, centers_new) = (centers_new, centers)
                if self._mini_batch_convergence(i, n_steps, n_samples, centers_squared_diff, batch_inertia):
                    break
        self.cluster_centers_ = centers
        self._n_features_out = self.cluster_centers_.shape[0]
        self.n_steps_ = i + 1
        self.n_iter_ = int(np.ceil((i + 1) * self._batch_size / n_samples))
        if self.compute_labels:
            (self.labels_, self.inertia_) = _labels_inertia_threadpool_limit(X, sample_weight, self.cluster_centers_, n_threads=self._n_threads)
        else:
            self.inertia_ = self._ewa_inertia * n_samples
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None, sample_weight=None):
        if False:
            return 10
        "Update k means estimate on a single mini-batch X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training instances to cluster. It must be noted that the data\n            will be converted to C ordering, which will cause a memory copy\n            if the given data is not C-contiguous.\n            If a sparse matrix is passed, a copy will be made if it's not in\n            CSR format.\n\n        y : Ignored\n            Not used, present here for API consistency by convention.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            The weights for each observation in X. If None, all observations\n            are assigned equal weight. `sample_weight` is not used during\n            initialization if `init` is a callable or a user provided array.\n\n        Returns\n        -------\n        self : object\n            Return updated estimator.\n        "
        has_centers = hasattr(self, 'cluster_centers_')
        X = self._validate_data(X, accept_sparse='csr', dtype=[np.float64, np.float32], order='C', accept_large_sparse=False, reset=not has_centers)
        self._random_state = getattr(self, '_random_state', check_random_state(self.random_state))
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self.n_steps_ = getattr(self, 'n_steps_', 0)
        x_squared_norms = row_norms(X, squared=True)
        if not has_centers:
            self._check_params_vs_input(X)
            self._n_threads = _openmp_effective_n_threads()
            init = self.init
            if _is_arraylike_not_scalar(init):
                init = check_array(init, dtype=X.dtype, copy=True, order='C')
                self._validate_center_shape(X, init)
            self._check_mkl_vcomp(X, X.shape[0])
            self.cluster_centers_ = self._init_centroids(X, x_squared_norms=x_squared_norms, init=init, random_state=self._random_state, init_size=self._init_size, sample_weight=sample_weight)
            self._counts = np.zeros(self.n_clusters, dtype=X.dtype)
            self._n_since_last_reassign = 0
        with threadpool_limits(limits=1, user_api='blas'):
            _mini_batch_step(X, sample_weight=sample_weight, centers=self.cluster_centers_, centers_new=self.cluster_centers_, weight_sums=self._counts, random_state=self._random_state, random_reassign=self._random_reassign(), reassignment_ratio=self.reassignment_ratio, verbose=self.verbose, n_threads=self._n_threads)
        if self.compute_labels:
            (self.labels_, self.inertia_) = _labels_inertia_threadpool_limit(X, sample_weight, self.cluster_centers_, n_threads=self._n_threads)
        self.n_steps_ += 1
        self._n_features_out = self.cluster_centers_.shape[0]
        return self