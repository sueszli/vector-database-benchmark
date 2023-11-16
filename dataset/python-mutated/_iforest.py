import numbers
from numbers import Integral, Real
from warnings import warn
import numpy as np
from scipy.sparse import issparse
from ..base import OutlierMixin, _fit_context
from ..tree import ExtraTreeRegressor
from ..tree._tree import DTYPE as tree_dtype
from ..utils import check_array, check_random_state, gen_batches, get_chunk_n_rows
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils.validation import _num_samples, check_is_fitted
from ._bagging import BaseBagging
__all__ = ['IsolationForest']

class IsolationForest(OutlierMixin, BaseBagging):
    """
    Isolation Forest Algorithm.

    Return the anomaly score of each sample using the IsolationForest algorithm

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Read more in the :ref:`User Guide <isolation_forest>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].

        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max(1, int(max_features * n_features_in_))` features.

        Note: using a float number less than 1.0 or integer less than number of
        features will enable feature subsampling and leads to a longer runtime.

    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        Controls the verbosity of the tree building process.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.21

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.ExtraTreeRegressor` instance
        The child estimator template used to create the collection of
        fitted sub-estimators.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    base_estimator_ : ExtraTreeRegressor instance
        The child estimator template used to create the collection of
        fitted sub-estimators.

        .. deprecated:: 1.2
            `base_estimator_` is deprecated and will be removed in 1.4.
            Use `estimator_` instead.

    estimators_ : list of ExtraTreeRegressor instances
        The collection of fitted sub-estimators.

    estimators_features_ : list of ndarray
        The subset of drawn features for each base estimator.

    estimators_samples_ : list of ndarray
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : int
        The actual number of samples.

    offset_ : float
        Offset used to define the decision function from the raw scores. We
        have the relation: ``decision_function = score_samples - offset_``.
        ``offset_`` is defined as follows. When the contamination parameter is
        set to "auto", the offset is equal to -0.5 as the scores of inliers are
        close to 0 and the scores of outliers are close to -1. When a
        contamination parameter different than "auto" is provided, the offset
        is defined in such a way we obtain the expected number of outliers
        (samples with decision function < 0) in training.

        .. versionadded:: 0.20

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.covariance.EllipticEnvelope : An object for detecting outliers in a
        Gaussian distributed dataset.
    sklearn.svm.OneClassSVM : Unsupervised Outlier Detection.
        Estimate the support of a high-dimensional distribution.
        The implementation is based on libsvm.
    sklearn.neighbors.LocalOutlierFactor : Unsupervised Outlier Detection
        using Local Outlier Factor (LOF).

    Notes
    -----
    The implementation is based on an ensemble of ExtraTreeRegressor. The
    maximum depth of each tree is set to ``ceil(log_2(n))`` where
    :math:`n` is the number of samples used to build the tree
    (see (Liu et al., 2008) for more details).

    References
    ----------
    .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
           Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
    .. [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based
           anomaly detection." ACM Transactions on Knowledge Discovery from
           Data (TKDD) 6.1 (2012): 3.

    Examples
    --------
    >>> from sklearn.ensemble import IsolationForest
    >>> X = [[-1.1], [0.3], [0.5], [100]]
    >>> clf = IsolationForest(random_state=0).fit(X)
    >>> clf.predict([[0.1], [0], [90]])
    array([ 1,  1, -1])

    For an example of using isolation forest for anomaly detection see
    :ref:`sphx_glr_auto_examples_ensemble_plot_isolation_forest.py`.
    """
    _parameter_constraints: dict = {'n_estimators': [Interval(Integral, 1, None, closed='left')], 'max_samples': [StrOptions({'auto'}), Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='right')], 'contamination': [StrOptions({'auto'}), Interval(Real, 0, 0.5, closed='right')], 'max_features': [Integral, Interval(Real, 0, 1, closed='right')], 'bootstrap': ['boolean'], 'n_jobs': [Integral, None], 'random_state': ['random_state'], 'verbose': ['verbose'], 'warm_start': ['boolean']}

    def __init__(self, *, n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):
        if False:
            while True:
                i = 10
        super().__init__(estimator=ExtraTreeRegressor(max_features=1, splitter='random', random_state=random_state), bootstrap=bootstrap, bootstrap_features=False, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, warm_start=warm_start, n_jobs=n_jobs, random_state=random_state, verbose=verbose)
        self.contamination = contamination

    def _set_oob_score(self, X, y):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('OOB score not supported by iforest')

    def _parallel_args(self):
        if False:
            while True:
                i = 10
        return {'prefer': 'threads'}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        if False:
            while True:
                i = 10
        '\n        Fit estimator.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Use ``dtype=np.float32`` for maximum\n            efficiency. Sparse matrices are also supported, use sparse\n            ``csc_matrix`` for maximum efficiency.\n\n        y : Ignored\n            Not used, present for API consistency by convention.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights. If None, then samples are equally weighted.\n\n        Returns\n        -------\n        self : object\n            Fitted estimator.\n        '
        X = self._validate_data(X, accept_sparse=['csc'], dtype=tree_dtype)
        if issparse(X):
            X.sort_indices()
        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])
        n_samples = X.shape[0]
        if isinstance(self.max_samples, str) and self.max_samples == 'auto':
            max_samples = min(256, n_samples)
        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn('max_samples (%s) is greater than the total number of samples (%s). max_samples will be set to n_samples for estimation.' % (self.max_samples, n_samples))
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:
            max_samples = int(self.max_samples * X.shape[0])
        self.max_samples_ = max_samples
        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        super()._fit(X, y, max_samples, max_depth=max_depth, sample_weight=sample_weight, check_input=False)
        (self._average_path_length_per_tree, self._decision_path_lengths) = zip(*[(_average_path_length(tree.tree_.n_node_samples), tree.tree_.compute_node_depths()) for tree in self.estimators_])
        if self.contamination == 'auto':
            self.offset_ = -0.5
            return self
        self.offset_ = np.percentile(self._score_samples(X), 100.0 * self.contamination)
        return self

    def predict(self, X):
        if False:
            return 10
        '\n        Predict if a particular sample is an outlier or not.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        Returns\n        -------\n        is_inlier : ndarray of shape (n_samples,)\n            For each observation, tells whether or not (+1 or -1) it should\n            be considered as an inlier according to the fitted model.\n        '
        check_is_fitted(self)
        decision_func = self.decision_function(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < 0] = -1
        return is_inlier

    def decision_function(self, X):
        if False:
            for i in range(10):
                print('nop')
        '\n        Average anomaly score of X of the base classifiers.\n\n        The anomaly score of an input sample is computed as\n        the mean anomaly score of the trees in the forest.\n\n        The measure of normality of an observation given a tree is the depth\n        of the leaf containing this observation, which is equivalent to\n        the number of splittings required to isolate this point. In case of\n        several observations n_left in the leaf, the average path length of\n        a n_left samples isolation tree is added.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        Returns\n        -------\n        scores : ndarray of shape (n_samples,)\n            The anomaly score of the input samples.\n            The lower, the more abnormal. Negative scores represent outliers,\n            positive scores represent inliers.\n        '
        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        if False:
            i = 10
            return i + 15
        '\n        Opposite of the anomaly score defined in the original paper.\n\n        The anomaly score of an input sample is computed as\n        the mean anomaly score of the trees in the forest.\n\n        The measure of normality of an observation given a tree is the depth\n        of the leaf containing this observation, which is equivalent to\n        the number of splittings required to isolate this point. In case of\n        several observations n_left in the leaf, the average path length of\n        a n_left samples isolation tree is added.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples.\n\n        Returns\n        -------\n        scores : ndarray of shape (n_samples,)\n            The anomaly score of the input samples.\n            The lower, the more abnormal.\n        '
        X = self._validate_data(X, accept_sparse='csr', dtype=np.float32, reset=False)
        return self._score_samples(X)

    def _score_samples(self, X):
        if False:
            i = 10
            return i + 15
        'Private version of score_samples without input validation.\n\n        Input validation would remove feature names, so we disable it.\n        '
        check_is_fitted(self)
        return -self._compute_chunked_score_samples(X)

    def _compute_chunked_score_samples(self, X):
        if False:
            for i in range(10):
                print('nop')
        n_samples = _num_samples(X)
        if self._max_features == X.shape[1]:
            subsample_features = False
        else:
            subsample_features = True
        chunk_n_rows = get_chunk_n_rows(row_bytes=16 * self._max_features, max_n_rows=n_samples)
        slices = gen_batches(n_samples, chunk_n_rows)
        scores = np.zeros(n_samples, order='f')
        for sl in slices:
            scores[sl] = self._compute_score_samples(X[sl], subsample_features)
        return scores

    def _compute_score_samples(self, X, subsample_features):
        if False:
            while True:
                i = 10
        '\n        Compute the score of each samples in X going through the extra trees.\n\n        Parameters\n        ----------\n        X : array-like or sparse matrix\n            Data matrix.\n\n        subsample_features : bool\n            Whether features should be subsampled.\n        '
        n_samples = X.shape[0]
        depths = np.zeros(n_samples, order='f')
        average_path_length_max_samples = _average_path_length([self._max_samples])
        for (tree_idx, (tree, features)) in enumerate(zip(self.estimators_, self.estimators_features_)):
            X_subset = X[:, features] if subsample_features else X
            leaves_index = tree.apply(X_subset, check_input=False)
            depths += self._decision_path_lengths[tree_idx][leaves_index] + self._average_path_length_per_tree[tree_idx][leaves_index] - 1.0
        denominator = len(self.estimators_) * average_path_length_max_samples
        scores = 2 ** (-np.divide(depths, denominator, out=np.ones_like(depths), where=denominator != 0))
        return scores

    def _more_tags(self):
        if False:
            for i in range(10):
                print('nop')
        return {'_xfail_checks': {'check_sample_weights_invariance': 'zero sample_weight is not equivalent to removing samples'}}

def _average_path_length(n_samples_leaf):
    if False:
        while True:
            i = 10
    '\n    The average path length in a n_samples iTree, which is equal to\n    the average path length of an unsuccessful BST search since the\n    latter has the same structure as an isolation tree.\n    Parameters\n    ----------\n    n_samples_leaf : array-like of shape (n_samples,)\n        The number of training samples in each test sample leaf, for\n        each estimators.\n\n    Returns\n    -------\n    average_path_length : ndarray of shape (n_samples,)\n    '
    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)
    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)
    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)
    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = 2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma) - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    return average_path_length.reshape(n_samples_leaf_shape)