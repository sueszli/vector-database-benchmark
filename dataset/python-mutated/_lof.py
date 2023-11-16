import warnings
from numbers import Real
import numpy as np
from ..base import OutlierMixin, _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ..utils.metaestimators import available_if
from ..utils.validation import check_is_fitted
from ._base import KNeighborsMixin, NeighborsBase
__all__ = ['LocalOutlierFactor']

class LocalOutlierFactor(KNeighborsMixin, OutlierMixin, NeighborsBase):
    """Unsupervised Outlier Detection using the Local Outlier Factor (LOF).

    The anomaly score of each sample is called the Local Outlier Factor.
    It measures the local deviation of the density of a given sample with respect
    to its neighbors.
    It is local in that the anomaly score depends on how isolated the object
    is with respect to the surrounding neighborhood.
    More precisely, locality is given by k-nearest neighbors, whose distance
    is used to estimate the local density.
    By comparing the local density of a sample to the local densities of its
    neighbors, one can identify samples that have a substantially lower density
    than their neighbors. These are considered outliers.

    .. versionadded:: 0.19

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf is size passed to :class:`BallTree` or :class:`KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

    p : float, default=2
        Parameter for the Minkowski metric from
        :func:`sklearn.metrics.pairwise_distances`. When p = 1, this
        is equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. When fitting this is used to define the
        threshold on the scores of the samples.

        - if 'auto', the threshold is determined as in the
          original paper,
        - if a float, the contamination should be in the range (0, 0.5].

        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.

    novelty : bool, default=False
        By default, LocalOutlierFactor is only meant to be used for outlier
        detection (novelty=False). Set novelty to True if you want to use
        LocalOutlierFactor for novelty detection. In this case be aware that
        you should only use predict, decision_function and score_samples
        on new unseen data and not on the training set; and note that the
        results obtained this way may differ from the standard LOF results.

        .. versionadded:: 0.20

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    negative_outlier_factor_ : ndarray of shape (n_samples,)
        The opposite LOF of the training samples. The higher, the more normal.
        Inliers tend to have a LOF score close to 1
        (``negative_outlier_factor_`` close to -1), while outliers tend to have
        a larger LOF score.

        The local outlier factor (LOF) of a sample captures its
        supposed 'degree of abnormality'.
        It is the average of the ratio of the local reachability density of
        a sample and those of its k-nearest neighbors.

    n_neighbors_ : int
        The actual number of neighbors used for :meth:`kneighbors` queries.

    offset_ : float
        Offset used to obtain binary labels from the raw scores.
        Observations having a negative_outlier_factor smaller than `offset_`
        are detected as abnormal.
        The offset is set to -1.5 (inliers score around -1), except when a
        contamination parameter different than "auto" is provided. In that
        case, the offset is defined in such a way we obtain the expected
        number of outliers in training.

        .. versionadded:: 0.20

    effective_metric_ : str
        The effective metric used for the distance computation.

    effective_metric_params_ : dict
        The effective additional keyword arguments for the metric function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_fit_ : int
        It is the number of samples in the fitted data.

    See Also
    --------
    sklearn.svm.OneClassSVM: Unsupervised Outlier Detection using
        Support Vector Machine.

    References
    ----------
    .. [1] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000, May).
           LOF: identifying density-based local outliers. In ACM sigmod record.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.neighbors import LocalOutlierFactor
    >>> X = [[-1.1], [0.2], [101.1], [0.3]]
    >>> clf = LocalOutlierFactor(n_neighbors=2)
    >>> clf.fit_predict(X)
    array([ 1,  1, -1,  1])
    >>> clf.negative_outlier_factor_
    array([ -0.9821...,  -1.0370..., -73.3697...,  -0.9821...])
    """
    _parameter_constraints: dict = {**NeighborsBase._parameter_constraints, 'contamination': [StrOptions({'auto'}), Interval(Real, 0, 0.5, closed='right')], 'novelty': ['boolean']}
    _parameter_constraints.pop('radius')

    def __init__(self, n_neighbors=20, *, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination='auto', novelty=False, n_jobs=None):
        if False:
            print('Hello World!')
        super().__init__(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs)
        self.contamination = contamination
        self.novelty = novelty

    def _check_novelty_fit_predict(self):
        if False:
            while True:
                i = 10
        if self.novelty:
            msg = 'fit_predict is not available when novelty=True. Use novelty=False if you want to predict on the training set.'
            raise AttributeError(msg)
        return True

    @available_if(_check_novelty_fit_predict)
    def fit_predict(self, X, y=None):
        if False:
            return 10
        'Fit the model to the training set X and return the labels.\n\n        **Not available for novelty detection (when novelty is set to True).**\n        Label is 1 for an inlier and -1 for an outlier according to the LOF\n        score and the contamination parameter.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features), default=None\n            The query sample or samples to compute the Local Outlier Factor\n            w.r.t. the training samples.\n\n        y : Ignored\n            Not used, present for API consistency by convention.\n\n        Returns\n        -------\n        is_inlier : ndarray of shape (n_samples,)\n            Returns -1 for anomalies/outliers and 1 for inliers.\n        '
        return self.fit(X)._predict()

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None):
        if False:
            print('Hello World!')
        "Fit the local outlier factor detector from the training dataset.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features) or                 (n_samples, n_samples) if metric='precomputed'\n            Training data.\n\n        y : Ignored\n            Not used, present for API consistency by convention.\n\n        Returns\n        -------\n        self : LocalOutlierFactor\n            The fitted local outlier factor detector.\n        "
        self._fit(X)
        n_samples = self.n_samples_fit_
        if self.n_neighbors > n_samples:
            warnings.warn('n_neighbors (%s) is greater than the total number of samples (%s). n_neighbors will be set to (n_samples - 1) for estimation.' % (self.n_neighbors, n_samples))
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))
        (self._distances_fit_X_, _neighbors_indices_fit_X_) = self.kneighbors(n_neighbors=self.n_neighbors_)
        if self._fit_X.dtype == np.float32:
            self._distances_fit_X_ = self._distances_fit_X_.astype(self._fit_X.dtype, copy=False)
        self._lrd = self._local_reachability_density(self._distances_fit_X_, _neighbors_indices_fit_X_)
        lrd_ratios_array = self._lrd[_neighbors_indices_fit_X_] / self._lrd[:, np.newaxis]
        self.negative_outlier_factor_ = -np.mean(lrd_ratios_array, axis=1)
        if self.contamination == 'auto':
            self.offset_ = -1.5
        else:
            self.offset_ = np.percentile(self.negative_outlier_factor_, 100.0 * self.contamination)
        return self

    def _check_novelty_predict(self):
        if False:
            print('Hello World!')
        if not self.novelty:
            msg = 'predict is not available when novelty=False, use fit_predict if you want to predict on training data. Use novelty=True if you want to use LOF for novelty detection and predict on new unseen data.'
            raise AttributeError(msg)
        return True

    @available_if(_check_novelty_predict)
    def predict(self, X=None):
        if False:
            return 10
        'Predict the labels (1 inlier, -1 outlier) of X according to LOF.\n\n        **Only available for novelty detection (when novelty is set to True).**\n        This method allows to generalize prediction to *new observations* (not\n        in the training set). Note that the result of ``clf.fit(X)`` then\n        ``clf.predict(X)`` with ``novelty=True`` may differ from the result\n        obtained by ``clf.fit_predict(X)`` with ``novelty=False``.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The query sample or samples to compute the Local Outlier Factor\n            w.r.t. the training samples.\n\n        Returns\n        -------\n        is_inlier : ndarray of shape (n_samples,)\n            Returns -1 for anomalies/outliers and +1 for inliers.\n        '
        return self._predict(X)

    def _predict(self, X=None):
        if False:
            while True:
                i = 10
        'Predict the labels (1 inlier, -1 outlier) of X according to LOF.\n\n        If X is None, returns the same as fit_predict(X_train).\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features), default=None\n            The query sample or samples to compute the Local Outlier Factor\n            w.r.t. the training samples. If None, makes prediction on the\n            training data without considering them as their own neighbors.\n\n        Returns\n        -------\n        is_inlier : ndarray of shape (n_samples,)\n            Returns -1 for anomalies/outliers and +1 for inliers.\n        '
        check_is_fitted(self)
        if X is not None:
            X = check_array(X, accept_sparse='csr')
            is_inlier = np.ones(X.shape[0], dtype=int)
            is_inlier[self.decision_function(X) < 0] = -1
        else:
            is_inlier = np.ones(self.n_samples_fit_, dtype=int)
            is_inlier[self.negative_outlier_factor_ < self.offset_] = -1
        return is_inlier

    def _check_novelty_decision_function(self):
        if False:
            return 10
        if not self.novelty:
            msg = 'decision_function is not available when novelty=False. Use novelty=True if you want to use LOF for novelty detection and compute decision_function for new unseen data. Note that the opposite LOF of the training samples is always available by considering the negative_outlier_factor_ attribute.'
            raise AttributeError(msg)
        return True

    @available_if(_check_novelty_decision_function)
    def decision_function(self, X):
        if False:
            print('Hello World!')
        'Shifted opposite of the Local Outlier Factor of X.\n\n        Bigger is better, i.e. large values correspond to inliers.\n\n        **Only available for novelty detection (when novelty is set to True).**\n        The shift offset allows a zero threshold for being an outlier.\n        The argument X is supposed to contain *new data*: if X contains a\n        point from training, it considers the later in its own neighborhood.\n        Also, the samples in X are not considered in the neighborhood of any\n        point.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The query sample or samples to compute the Local Outlier Factor\n            w.r.t. the training samples.\n\n        Returns\n        -------\n        shifted_opposite_lof_scores : ndarray of shape (n_samples,)\n            The shifted opposite of the Local Outlier Factor of each input\n            samples. The lower, the more abnormal. Negative scores represent\n            outliers, positive scores represent inliers.\n        '
        return self.score_samples(X) - self.offset_

    def _check_novelty_score_samples(self):
        if False:
            return 10
        if not self.novelty:
            msg = 'score_samples is not available when novelty=False. The scores of the training samples are always available through the negative_outlier_factor_ attribute. Use novelty=True if you want to use LOF for novelty detection and compute score_samples for new unseen data.'
            raise AttributeError(msg)
        return True

    @available_if(_check_novelty_score_samples)
    def score_samples(self, X):
        if False:
            print('Hello World!')
        'Opposite of the Local Outlier Factor of X.\n\n        It is the opposite as bigger is better, i.e. large values correspond\n        to inliers.\n\n        **Only available for novelty detection (when novelty is set to True).**\n        The argument X is supposed to contain *new data*: if X contains a\n        point from training, it considers the later in its own neighborhood.\n        Also, the samples in X are not considered in the neighborhood of any\n        point. Because of this, the scores obtained via ``score_samples`` may\n        differ from the standard LOF scores.\n        The standard LOF scores for the training data is available via the\n        ``negative_outlier_factor_`` attribute.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The query sample or samples to compute the Local Outlier Factor\n            w.r.t. the training samples.\n\n        Returns\n        -------\n        opposite_lof_scores : ndarray of shape (n_samples,)\n            The opposite of the Local Outlier Factor of each input samples.\n            The lower, the more abnormal.\n        '
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        (distances_X, neighbors_indices_X) = self.kneighbors(X, n_neighbors=self.n_neighbors_)
        if X.dtype == np.float32:
            distances_X = distances_X.astype(X.dtype, copy=False)
        X_lrd = self._local_reachability_density(distances_X, neighbors_indices_X)
        lrd_ratios_array = self._lrd[neighbors_indices_X] / X_lrd[:, np.newaxis]
        return -np.mean(lrd_ratios_array, axis=1)

    def _local_reachability_density(self, distances_X, neighbors_indices):
        if False:
            return 10
        'The local reachability density (LRD)\n\n        The LRD of a sample is the inverse of the average reachability\n        distance of its k-nearest neighbors.\n\n        Parameters\n        ----------\n        distances_X : ndarray of shape (n_queries, self.n_neighbors)\n            Distances to the neighbors (in the training samples `self._fit_X`)\n            of each query point to compute the LRD.\n\n        neighbors_indices : ndarray of shape (n_queries, self.n_neighbors)\n            Neighbors indices (of each query point) among training samples\n            self._fit_X.\n\n        Returns\n        -------\n        local_reachability_density : ndarray of shape (n_queries,)\n            The local reachability density of each sample.\n        '
        dist_k = self._distances_fit_X_[neighbors_indices, self.n_neighbors_ - 1]
        reach_dist_array = np.maximum(distances_X, dist_k)
        return 1.0 / (np.mean(reach_dist_array, axis=1) + 1e-10)

    def _more_tags(self):
        if False:
            return 10
        return {'preserves_dtype': [np.float64, np.float32]}