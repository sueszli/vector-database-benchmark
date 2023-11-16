"""IsolationForest Outlier Detector. Implemented on scikit-learn library.
"""
from __future__ import division
from __future__ import print_function
import numpy as np
from joblib import Parallel
from joblib.parallel import delayed
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from .base import BaseDetector
from ..utils.utility import invert_order

class IForest(BaseDetector):
    """Wrapper of scikit-learn Isolation Forest with more functionalities.

    The IsolationForest 'isolates' observations by randomly selecting a
    feature and then randomly selecting a split value between the maximum and
    minimum values of the selected feature.
    See :cite:`liu2008isolation,liu2012isolation` for details.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    behaviour : str, default='old'
        Behaviour of the ``decision_function`` which can be either 'old' or
        'new'. Passing ``behaviour='new'`` makes the ``decision_function``
        change to match other anomaly detection algorithm API which will be
        the default behaviour in the future. As explained in details in the
        ``offset_`` attribute documentation, the ``decision_function`` becomes
        dependent on the contamination parameter, in such a way that 0 becomes
        its natural threshold to detect outliers.

        .. versionadded:: 0.7.0
           ``behaviour`` is added in 0.7.0 for back-compatibility purpose.

        .. deprecated:: 0.20
           ``behaviour='old'`` is deprecated in sklearn 0.20 and will not be
           possible in 0.22.

        .. deprecated:: 0.22
           ``behaviour`` parameter will be deprecated in sklearn 0.22 and
           removed in 0.24.

        .. warning::
            Only applicable for sklearn 0.20 above.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : integer
        The actual number of samples

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0, bootstrap=False, n_jobs=1, behaviour='old', random_state=None, verbose=0):
        if False:
            for i in range(10):
                print('nop')
        super(IForest, self).__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.behaviour = behaviour
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        if False:
            print('Hello World!')
        'Fit detector. y is ignored in unsupervised methods.\n\n        Parameters\n        ----------\n        X : numpy array of shape (n_samples, n_features)\n            The input samples.\n\n        y : Ignored\n            Not used, present for API consistency by convention.\n\n        Returns\n        -------\n        self : object\n            Fitted estimator.\n        '
        X = check_array(X)
        self._set_n_classes(y)
        self.detector_ = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples, contamination=self.contamination, max_features=self.max_features, bootstrap=self.bootstrap, n_jobs=self.n_jobs, random_state=self.random_state, verbose=self.verbose)
        self.detector_.fit(X=X, y=None, sample_weight=None)
        self.decision_scores_ = invert_order(self.detector_.decision_function(X))
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        if False:
            print('Hello World!')
        'Predict raw anomaly score of X using the fitted detector.\n\n        The anomaly score of an input sample is computed based on different\n        detector algorithms. For consistency, outliers are assigned with\n        larger anomaly scores.\n\n        Parameters\n        ----------\n        X : numpy array of shape (n_samples, n_features)\n            The training input samples. Sparse matrices are accepted only\n            if they are supported by the base estimator.\n\n        Returns\n        -------\n        anomaly_scores : numpy array of shape (n_samples,)\n            The anomaly score of the input samples.\n        '
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        return invert_order(self.detector_.decision_function(X))

    @property
    def estimators_(self):
        if False:
            for i in range(10):
                print('nop')
        'The collection of fitted sub-estimators.\n        Decorator for scikit-learn Isolation Forest attributes.\n        '
        return self.detector_.estimators_

    @property
    def estimators_samples_(self):
        if False:
            while True:
                i = 10
        'The subset of drawn samples (i.e., the in-bag samples) for\n        each base estimator.\n        Decorator for scikit-learn Isolation Forest attributes.\n        '
        return self.detector_.estimators_samples_

    @property
    def max_samples_(self):
        if False:
            print('Hello World!')
        'The actual number of samples.\n        Decorator for scikit-learn Isolation Forest attributes.\n        '
        return self.detector_.max_samples_

    @property
    def feature_importances_(self):
        if False:
            print('Hello World!')
        'The impurity-based feature importance. The higher, the more\n        important the feature. The importance of a feature is computed as the\n        (normalized) total reduction of the criterion brought by that feature.\n        It is also known as the Gini importance.\n\n        .. warning::\n        impurity-based feature importance can be misleading for\n        high cardinality features (many unique values). See\n        https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html\n        as an alternative.\n\n        Returns\n        -------\n        feature_importances_ : ndarray of shape (n_features,)\n            The values of this array sum to 1, unless all trees are single node\n            trees consisting of only the root node, in which case it will be an\n            array of zeros.\n        '
        check_is_fitted(self)
        all_importances = Parallel(n_jobs=self.n_jobs)((delayed(getattr)(tree, 'feature_importances_') for tree in self.detector_.estimators_ if tree.tree_.node_count > 1))
        if not all_importances:
            return np.zeros(self.n_features_in_, dtype=np.float64)
        all_importances = np.mean(all_importances, axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)