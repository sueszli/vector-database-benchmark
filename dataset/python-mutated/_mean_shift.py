"""Mean shift clustering algorithm.

Mean shift clustering aims to discover *blobs* in a smooth density of
samples. It is a centroid based algorithm, which works by updating candidates
for centroids to be the mean of the points within a given region. These
candidates are then filtered in a post-processing stage to eliminate
near-duplicates to form the final set of centroids.

Seeding is performed using a binning technique for scalability.
"""
import warnings
from collections import defaultdict
from numbers import Integral, Real
import numpy as np
from .._config import config_context
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..metrics.pairwise import pairwise_distances_argmin
from ..neighbors import NearestNeighbors
from ..utils import check_array, check_random_state, gen_batches
from ..utils._param_validation import Interval, validate_params
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted

@validate_params({'X': ['array-like'], 'quantile': [Interval(Real, 0, 1, closed='both')], 'n_samples': [Interval(Integral, 1, None, closed='left'), None], 'random_state': ['random_state'], 'n_jobs': [Integral, None]}, prefer_skip_nested_validation=True)
def estimate_bandwidth(X, *, quantile=0.3, n_samples=None, random_state=0, n_jobs=None):
    if False:
        while True:
            i = 10
    'Estimate the bandwidth to use with the mean-shift algorithm.\n\n    This function takes time at least quadratic in `n_samples`. For large\n    datasets, it is wise to subsample by setting `n_samples`. Alternatively,\n    the parameter `bandwidth` can be set to a small value without estimating\n    it.\n\n    Parameters\n    ----------\n    X : array-like of shape (n_samples, n_features)\n        Input points.\n\n    quantile : float, default=0.3\n        Should be between [0, 1]\n        0.5 means that the median of all pairwise distances is used.\n\n    n_samples : int, default=None\n        The number of samples to use. If not given, all samples are used.\n\n    random_state : int, RandomState instance, default=None\n        The generator used to randomly select the samples from input points\n        for bandwidth estimation. Use an int to make the randomness\n        deterministic.\n        See :term:`Glossary <random_state>`.\n\n    n_jobs : int, default=None\n        The number of parallel jobs to run for neighbors search.\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    Returns\n    -------\n    bandwidth : float\n        The bandwidth parameter.\n    '
    X = check_array(X)
    random_state = check_random_state(random_state)
    if n_samples is not None:
        idx = random_state.permutation(X.shape[0])[:n_samples]
        X = X[idx]
    n_neighbors = int(X.shape[0] * quantile)
    if n_neighbors < 1:
        n_neighbors = 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
    nbrs.fit(X)
    bandwidth = 0.0
    for batch in gen_batches(len(X), 500):
        (d, _) = nbrs.kneighbors(X[batch, :], return_distance=True)
        bandwidth += np.max(d, axis=1).sum()
    return bandwidth / X.shape[0]

def _mean_shift_single_seed(my_mean, X, nbrs, max_iter):
    if False:
        print('Hello World!')
    bandwidth = nbrs.get_params()['radius']
    stop_thresh = 0.001 * bandwidth
    completed_iterations = 0
    while True:
        i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth, return_distance=False)[0]
        points_within = X[i_nbrs]
        if len(points_within) == 0:
            break
        my_old_mean = my_mean
        my_mean = np.mean(points_within, axis=0)
        if np.linalg.norm(my_mean - my_old_mean) < stop_thresh or completed_iterations == max_iter:
            break
        completed_iterations += 1
    return (tuple(my_mean), len(points_within), completed_iterations)

@validate_params({'X': ['array-like']}, prefer_skip_nested_validation=False)
def mean_shift(X, *, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, max_iter=300, n_jobs=None):
    if False:
        print('Hello World!')
    'Perform mean shift clustering of data using a flat kernel.\n\n    Read more in the :ref:`User Guide <mean_shift>`.\n\n    Parameters\n    ----------\n\n    X : array-like of shape (n_samples, n_features)\n        Input data.\n\n    bandwidth : float, default=None\n        Kernel bandwidth. If not None, must be in the range [0, +inf).\n\n        If None, the bandwidth is determined using a heuristic based on\n        the median of all pairwise distances. This will take quadratic time in\n        the number of samples. The sklearn.cluster.estimate_bandwidth function\n        can be used to do this more efficiently.\n\n    seeds : array-like of shape (n_seeds, n_features) or None\n        Point used as initial kernel locations. If None and bin_seeding=False,\n        each data point is used as a seed. If None and bin_seeding=True,\n        see bin_seeding.\n\n    bin_seeding : bool, default=False\n        If true, initial kernel locations are not locations of all\n        points, but rather the location of the discretized version of\n        points, where points are binned onto a grid whose coarseness\n        corresponds to the bandwidth. Setting this option to True will speed\n        up the algorithm because fewer seeds will be initialized.\n        Ignored if seeds argument is not None.\n\n    min_bin_freq : int, default=1\n       To speed up the algorithm, accept only those bins with at least\n       min_bin_freq points as seeds.\n\n    cluster_all : bool, default=True\n        If true, then all points are clustered, even those orphans that are\n        not within any kernel. Orphans are assigned to the nearest kernel.\n        If false, then orphans are given cluster label -1.\n\n    max_iter : int, default=300\n        Maximum number of iterations, per seed point before the clustering\n        operation terminates (for that seed point), if has not converged yet.\n\n    n_jobs : int, default=None\n        The number of jobs to use for the computation. The following tasks benefit\n        from the parallelization:\n\n        - The search of nearest neighbors for bandwidth estimation and label\n          assignments. See the details in the docstring of the\n          ``NearestNeighbors`` class.\n        - Hill-climbing optimization for all seeds.\n\n        See :term:`Glossary <n_jobs>` for more details.\n\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n        .. versionadded:: 0.17\n           Parallel Execution using *n_jobs*.\n\n    Returns\n    -------\n\n    cluster_centers : ndarray of shape (n_clusters, n_features)\n        Coordinates of cluster centers.\n\n    labels : ndarray of shape (n_samples,)\n        Cluster labels for each point.\n\n    Notes\n    -----\n    For an example, see :ref:`examples/cluster/plot_mean_shift.py\n    <sphx_glr_auto_examples_cluster_plot_mean_shift.py>`.\n    '
    model = MeanShift(bandwidth=bandwidth, seeds=seeds, min_bin_freq=min_bin_freq, bin_seeding=bin_seeding, cluster_all=cluster_all, n_jobs=n_jobs, max_iter=max_iter).fit(X)
    return (model.cluster_centers_, model.labels_)

def get_bin_seeds(X, bin_size, min_bin_freq=1):
    if False:
        i = 10
        return i + 15
    "Find seeds for mean_shift.\n\n    Finds seeds by first binning data onto a grid whose lines are\n    spaced bin_size apart, and then choosing those bins with at least\n    min_bin_freq points.\n\n    Parameters\n    ----------\n\n    X : array-like of shape (n_samples, n_features)\n        Input points, the same points that will be used in mean_shift.\n\n    bin_size : float\n        Controls the coarseness of the binning. Smaller values lead\n        to more seeding (which is computationally more expensive). If you're\n        not sure how to set this, set it to the value of the bandwidth used\n        in clustering.mean_shift.\n\n    min_bin_freq : int, default=1\n        Only bins with at least min_bin_freq will be selected as seeds.\n        Raising this value decreases the number of seeds found, which\n        makes mean_shift computationally cheaper.\n\n    Returns\n    -------\n    bin_seeds : array-like of shape (n_samples, n_features)\n        Points used as initial kernel positions in clustering.mean_shift.\n    "
    if bin_size == 0:
        return X
    bin_sizes = defaultdict(int)
    for point in X:
        binned_point = np.round(point / bin_size)
        bin_sizes[tuple(binned_point)] += 1
    bin_seeds = np.array([point for (point, freq) in bin_sizes.items() if freq >= min_bin_freq], dtype=np.float32)
    if len(bin_seeds) == len(X):
        warnings.warn('Binning data failed with provided bin_size=%f, using data points as seeds.' % bin_size)
        return X
    bin_seeds = bin_seeds * bin_size
    return bin_seeds

class MeanShift(ClusterMixin, BaseEstimator):
    """Mean shift clustering using a flat kernel.

    Mean shift clustering aims to discover "blobs" in a smooth density of
    samples. It is a centroid-based algorithm, which works by updating
    candidates for centroids to be the mean of the points within a given
    region. These candidates are then filtered in a post-processing stage to
    eliminate near-duplicates to form the final set of centroids.

    Seeding is performed using a binning technique for scalability.

    Read more in the :ref:`User Guide <mean_shift>`.

    Parameters
    ----------
    bandwidth : float, default=None
        Bandwidth used in the flat kernel.

        If not given, the bandwidth is estimated using
        sklearn.cluster.estimate_bandwidth; see the documentation for that
        function for hints on scalability (see also the Notes, below).

    seeds : array-like of shape (n_samples, n_features), default=None
        Seeds used to initialize kernels. If not set,
        the seeds are calculated by clustering.get_bin_seeds
        with bandwidth as the grid size and default values for
        other parameters.

    bin_seeding : bool, default=False
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        The default value is False.
        Ignored if seeds argument is not None.

    min_bin_freq : int, default=1
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds.

    cluster_all : bool, default=True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    n_jobs : int, default=None
        The number of jobs to use for the computation. The following tasks benefit
        from the parallelization:

        - The search of nearest neighbors for bandwidth estimation and label
          assignments. See the details in the docstring of the
          ``NearestNeighbors`` class.
        - Hill-climbing optimization for all seeds.

        See :term:`Glossary <n_jobs>` for more details.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    max_iter : int, default=300
        Maximum number of iterations, per seed point before the clustering
        operation terminates (for that seed point), if has not converged yet.

        .. versionadded:: 0.22

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    n_iter_ : int
        Maximum number of iterations performed on each seed.

        .. versionadded:: 0.22

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    KMeans : K-Means clustering.

    Notes
    -----

    Scalability:

    Because this implementation uses a flat kernel and
    a Ball Tree to look up members of each kernel, the complexity will tend
    towards O(T*n*log(n)) in lower dimensions, with n the number of samples
    and T the number of points. In higher dimensions the complexity will
    tend towards O(T*n^2).

    Scalability can be boosted by using fewer seeds, for example by using
    a higher value of min_bin_freq in the get_bin_seeds function.

    Note that the estimate_bandwidth function is much less scalable than the
    mean shift algorithm and will be the bottleneck if it is used.

    References
    ----------

    Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
    feature space analysis". IEEE Transactions on Pattern Analysis and
    Machine Intelligence. 2002. pp. 603-619.

    Examples
    --------
    >>> from sklearn.cluster import MeanShift
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = MeanShift(bandwidth=2).fit(X)
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])
    >>> clustering.predict([[0, 0], [5, 5]])
    array([1, 0])
    >>> clustering
    MeanShift(bandwidth=2)
    """
    _parameter_constraints: dict = {'bandwidth': [Interval(Real, 0, None, closed='neither'), None], 'seeds': ['array-like', None], 'bin_seeding': ['boolean'], 'min_bin_freq': [Interval(Integral, 1, None, closed='left')], 'cluster_all': ['boolean'], 'n_jobs': [Integral, None], 'max_iter': [Interval(Integral, 0, None, closed='left')]}

    def __init__(self, *, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300):
        if False:
            return 10
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.cluster_all = cluster_all
        self.min_bin_freq = min_bin_freq
        self.n_jobs = n_jobs
        self.max_iter = max_iter

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        'Perform clustering.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Samples to cluster.\n\n        y : Ignored\n            Not used, present for API consistency by convention.\n\n        Returns\n        -------\n        self : object\n               Fitted instance.\n        '
        X = self._validate_data(X)
        bandwidth = self.bandwidth
        if bandwidth is None:
            bandwidth = estimate_bandwidth(X, n_jobs=self.n_jobs)
        seeds = self.seeds
        if seeds is None:
            if self.bin_seeding:
                seeds = get_bin_seeds(X, bandwidth, self.min_bin_freq)
            else:
                seeds = X
        (n_samples, n_features) = X.shape
        center_intensity_dict = {}
        nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(X)
        all_res = Parallel(n_jobs=self.n_jobs)((delayed(_mean_shift_single_seed)(seed, X, nbrs, self.max_iter) for seed in seeds))
        for i in range(len(seeds)):
            if all_res[i][1]:
                center_intensity_dict[all_res[i][0]] = all_res[i][1]
        self.n_iter_ = max([x[2] for x in all_res])
        if not center_intensity_dict:
            raise ValueError('No point was within bandwidth=%f of any seed. Try a different seeding strategy                              or increase the bandwidth.' % bandwidth)
        sorted_by_intensity = sorted(center_intensity_dict.items(), key=lambda tup: (tup[1], tup[0]), reverse=True)
        sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
        unique = np.ones(len(sorted_centers), dtype=bool)
        nbrs = NearestNeighbors(radius=bandwidth, n_jobs=self.n_jobs).fit(sorted_centers)
        for (i, center) in enumerate(sorted_centers):
            if unique[i]:
                neighbor_idxs = nbrs.radius_neighbors([center], return_distance=False)[0]
                unique[neighbor_idxs] = 0
                unique[i] = 1
        cluster_centers = sorted_centers[unique]
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs).fit(cluster_centers)
        labels = np.zeros(n_samples, dtype=int)
        (distances, idxs) = nbrs.kneighbors(X)
        if self.cluster_all:
            labels = idxs.flatten()
        else:
            labels.fill(-1)
            bool_selector = distances.flatten() <= bandwidth
            labels[bool_selector] = idxs.flatten()[bool_selector]
        (self.cluster_centers_, self.labels_) = (cluster_centers, labels)
        return self

    def predict(self, X):
        if False:
            print('Hello World!')
        'Predict the closest cluster each sample in X belongs to.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            New data to predict.\n\n        Returns\n        -------\n        labels : ndarray of shape (n_samples,)\n            Index of the cluster each sample belongs to.\n        '
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        with config_context(assume_finite=True):
            return pairwise_distances_argmin(X, self.cluster_centers_)