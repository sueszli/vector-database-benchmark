"""Weight Boosting.

This module contains weight boosting estimators for both classification and
regression.

The module structure is the following:

- The `BaseWeightBoosting` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ from each other in the loss function that is optimized.

- :class:`~sklearn.ensemble.AdaBoostClassifier` implements adaptive boosting
  (AdaBoost-SAMME) for classification problems.

- :class:`~sklearn.ensemble.AdaBoostRegressor` implements adaptive boosting
  (AdaBoost.R2) for regression problems.
"""
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import xlogy
from ..base import ClassifierMixin, RegressorMixin, _fit_context, is_classifier, is_regressor
from ..metrics import accuracy_score, r2_score
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import _safe_indexing, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.extmath import softmax, stable_cumsum
from ..utils.metadata_routing import _raise_for_unsupported_routing, _RoutingNotSupportedMixin
from ..utils.validation import _check_sample_weight, _num_samples, check_is_fitted, has_fit_parameter
from ._base import BaseEnsemble
__all__ = ['AdaBoostClassifier', 'AdaBoostRegressor']

class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    _parameter_constraints: dict = {'estimator': [HasMethods(['fit', 'predict']), None], 'n_estimators': [Interval(Integral, 1, None, closed='left')], 'learning_rate': [Interval(Real, 0, None, closed='neither')], 'random_state': ['random_state'], 'base_estimator': [HasMethods(['fit', 'predict']), StrOptions({'deprecated'}), None]}

    @abstractmethod
    def __init__(self, estimator=None, *, n_estimators=50, estimator_params=tuple(), learning_rate=1.0, random_state=None, base_estimator='deprecated'):
        if False:
            print('Hello World!')
        super().__init__(estimator=estimator, n_estimators=n_estimators, estimator_params=estimator_params, base_estimator=base_estimator)
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _check_X(self, X):
        if False:
            while True:
                i = 10
        return self._validate_data(X, accept_sparse=['csr', 'csc'], ensure_2d=True, allow_nd=True, dtype=None, reset=False)

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, sample_weight=None):
        if False:
            return 10
        'Build a boosted classifier/regressor from the training set (X, y).\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples. Sparse matrix can be CSC, CSR, COO,\n            DOK, or LIL. COO, DOK, and LIL are converted to CSR.\n\n        y : array-like of shape (n_samples,)\n            The target values.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights. If None, the sample weights are initialized to\n            1 / n_samples.\n\n        Returns\n        -------\n        self : object\n            Fitted estimator.\n        '
        _raise_for_unsupported_routing(self, 'fit', sample_weight=sample_weight)
        (X, y) = self._validate_data(X, y, accept_sparse=['csr', 'csc'], ensure_2d=True, allow_nd=True, dtype=None, y_numeric=is_regressor(self))
        sample_weight = _check_sample_weight(sample_weight, X, np.float64, copy=True, only_non_negative=True)
        sample_weight /= sample_weight.sum()
        self._validate_estimator()
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        random_state = check_random_state(self.random_state)
        epsilon = np.finfo(sample_weight.dtype).eps
        zero_weight_mask = sample_weight == 0.0
        for iboost in range(self.n_estimators):
            sample_weight = np.clip(sample_weight, a_min=epsilon, a_max=None)
            sample_weight[zero_weight_mask] = 0.0
            (sample_weight, estimator_weight, estimator_error) = self._boost(iboost, X, y, sample_weight, random_state)
            if sample_weight is None:
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error
            if estimator_error == 0:
                break
            sample_weight_sum = np.sum(sample_weight)
            if not np.isfinite(sample_weight_sum):
                warnings.warn(f'Sample weights have reached infinite values, at iteration {iboost}, causing overflow. Iterations stopped. Try lowering the learning rate.', stacklevel=2)
                break
            if sample_weight_sum <= 0:
                break
            if iboost < self.n_estimators - 1:
                sample_weight /= sample_weight_sum
        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        if False:
            while True:
                i = 10
        'Implement a single boost.\n\n        Warning: This method needs to be overridden by subclasses.\n\n        Parameters\n        ----------\n        iboost : int\n            The index of the current boost iteration.\n\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples. Sparse matrix can be CSC, CSR, COO,\n            DOK, or LIL. COO, DOK, and LIL are converted to CSR.\n\n        y : array-like of shape (n_samples,)\n            The target values (class labels).\n\n        sample_weight : array-like of shape (n_samples,)\n            The current sample weights.\n\n        random_state : RandomState\n            The current random number generator\n\n        Returns\n        -------\n        sample_weight : array-like of shape (n_samples,) or None\n            The reweighted sample weights.\n            If None then boosting has terminated early.\n\n        estimator_weight : float\n            The weight for the current boost.\n            If None then boosting has terminated early.\n\n        error : float\n            The classification error for the current boost.\n            If None then boosting has terminated early.\n        '
        pass

    def staged_score(self, X, y, sample_weight=None):
        if False:
            print('Hello World!')
        'Return staged scores for X, y.\n\n        This generator method yields the ensemble score after each iteration of\n        boosting and therefore allows monitoring, such as to determine the\n        score on a test set after each boost.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples. Sparse matrix can be CSC, CSR, COO,\n            DOK, or LIL. COO, DOK, and LIL are converted to CSR.\n\n        y : array-like of shape (n_samples,)\n            Labels for X.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights.\n\n        Yields\n        ------\n        z : float\n        '
        X = self._check_X(X)
        for y_pred in self.staged_predict(X):
            if is_classifier(self):
                yield accuracy_score(y, y_pred, sample_weight=sample_weight)
            else:
                yield r2_score(y, y_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        if False:
            print('Hello World!')
        'The impurity-based feature importances.\n\n        The higher, the more important the feature.\n        The importance of a feature is computed as the (normalized)\n        total reduction of the criterion brought by that feature.  It is also\n        known as the Gini importance.\n\n        Warning: impurity-based feature importances can be misleading for\n        high cardinality features (many unique values). See\n        :func:`sklearn.inspection.permutation_importance` as an alternative.\n\n        Returns\n        -------\n        feature_importances_ : ndarray of shape (n_features,)\n            The feature importances.\n        '
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError('Estimator not fitted, call `fit` before `feature_importances_`.')
        try:
            norm = self.estimator_weights_.sum()
            return sum((weight * clf.feature_importances_ for (weight, clf) in zip(self.estimator_weights_, self.estimators_))) / norm
        except AttributeError as e:
            raise AttributeError('Unable to compute feature importances since estimator does not have a feature_importances_ attribute') from e

def _samme_proba(estimator, n_classes, X):
    if False:
        while True:
            i = 10
    'Calculate algorithm 4, step 2, equation c) of Zhu et al [1].\n\n    References\n    ----------\n    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.\n\n    '
    proba = estimator.predict_proba(X)
    np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
    log_proba = np.log(proba)
    return (n_classes - 1) * (log_proba - 1.0 / n_classes * log_proba.sum(axis=1)[:, np.newaxis])

class AdaBoostClassifier(_RoutingNotSupportedMixin, ClassifierMixin, BaseWeightBoosting):
    """An AdaBoost classifier.

    An AdaBoost [1]_ classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    This class implements the algorithm based on [2]_.

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
        initialized with `max_depth=1`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration. A higher
        learning rate increases the contribution of each classifier. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`.

    algorithm : {'SAMME', 'SAMME.R'}, default='SAMME.R'
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

        .. deprecated:: 1.4
            `"SAMME.R"` is deprecated and will be removed in version 1.6.
            '"SAMME"' will become the default.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `estimator` at each
        boosting iteration.
        Thus, it is only used when `estimator` exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    base_estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
        initialized with `max_depth=1`.

        .. deprecated:: 1.2
            `base_estimator` is deprecated and will be removed in 1.4.
            Use `estimator` instead.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. deprecated:: 1.2
            `base_estimator_` is deprecated and will be removed in 1.4.
            Use `estimator_` instead.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances if supported by the
        ``estimator`` (when based on decision trees).

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdaBoostRegressor : An AdaBoost regressor that begins by fitting a
        regressor on the original dataset and then fits additional copies of
        the regressor on the same dataset but where the weights of instances
        are adjusted according to the error of the current prediction.

    GradientBoostingClassifier : GB builds an additive model in a forward
        stage-wise fashion. Regression trees are fit on the negative gradient
        of the binomial or multinomial deviance loss function. Binary
        classification is a special case where only a single regression tree is
        induced.

    sklearn.tree.DecisionTreeClassifier : A non-parametric supervised learning
        method used for classification.
        Creates a model that predicts the value of a target variable by
        learning simple decision rules inferred from the data features.

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] :doi:`J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class adaboost."
           Statistics and its Interface 2.3 (2009): 349-360.
           <10.4310/SII.2009.v2.n3.a8>`

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)
    >>> clf.fit(X, y)
    AdaBoostClassifier(algorithm='SAMME', n_estimators=100, random_state=0)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    >>> clf.score(X, y)
    0.96...
    """
    _parameter_constraints: dict = {**BaseWeightBoosting._parameter_constraints, 'algorithm': [StrOptions({'SAMME', 'SAMME.R'})]}

    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None, base_estimator='deprecated'):
        if False:
            while True:
                i = 10
        super().__init__(estimator=estimator, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, base_estimator=base_estimator)
        self.algorithm = algorithm

    def _validate_estimator(self):
        if False:
            return 10
        'Check the estimator and set the estimator_ attribute.'
        super()._validate_estimator(default=DecisionTreeClassifier(max_depth=1))
        if self.algorithm != 'SAMME':
            warnings.warn('The SAMME.R algorithm (the default) is deprecated and will be removed in 1.6. Use the SAMME algorithm to circumvent this warning.', FutureWarning)
            if not hasattr(self.estimator_, 'predict_proba'):
                raise TypeError("AdaBoostClassifier with algorithm='SAMME.R' requires that the weak learner supports the calculation of class probabilities with a predict_proba method.\nPlease change the base estimator or set algorithm='SAMME' instead.")
        if not has_fit_parameter(self.estimator_, 'sample_weight'):
            raise ValueError(f"{self.estimator.__class__.__name__} doesn't support sample_weight.")

    def _boost(self, iboost, X, y, sample_weight, random_state):
        if False:
            i = 10
            return i + 15
        'Implement a single boost.\n\n        Perform a single boost according to the real multi-class SAMME.R\n        algorithm or to the discrete SAMME algorithm and return the updated\n        sample weights.\n\n        Parameters\n        ----------\n        iboost : int\n            The index of the current boost iteration.\n\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples.\n\n        y : array-like of shape (n_samples,)\n            The target values (class labels).\n\n        sample_weight : array-like of shape (n_samples,)\n            The current sample weights.\n\n        random_state : RandomState instance\n            The RandomState instance used if the base estimator accepts a\n            `random_state` attribute.\n\n        Returns\n        -------\n        sample_weight : array-like of shape (n_samples,) or None\n            The reweighted sample weights.\n            If None then boosting has terminated early.\n\n        estimator_weight : float\n            The weight for the current boost.\n            If None then boosting has terminated early.\n\n        estimator_error : float\n            The classification error for the current boost.\n            If None then boosting has terminated early.\n        '
        if self.algorithm == 'SAMME.R':
            return self._boost_real(iboost, X, y, sample_weight, random_state)
        else:
            return self._boost_discrete(iboost, X, y, sample_weight, random_state)

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        if False:
            for i in range(10):
                print('nop')
        'Implement a single boost using the SAMME.R real algorithm.'
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict_proba = estimator.predict_proba(X)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)
        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)
        incorrect = y_predict != y
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        if estimator_error <= 0:
            return (sample_weight, 1.0, 0.0)
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
        y_coding = y_codes.take(classes == y[:, np.newaxis])
        proba = y_predict_proba
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
        estimator_weight = -1.0 * self.learning_rate * ((n_classes - 1.0) / n_classes) * xlogy(y_coding, y_predict_proba).sum(axis=1)
        if not iboost == self.n_estimators - 1:
            sample_weight *= np.exp(estimator_weight * ((sample_weight > 0) | (estimator_weight < 0)))
        return (sample_weight, 1.0, estimator_error)

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        if False:
            return 10
        'Implement a single boost using the SAMME discrete algorithm.'
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)
        incorrect = y_predict != y
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        if estimator_error <= 0:
            return (sample_weight, 1.0, 0.0)
        n_classes = self.n_classes_
        if estimator_error >= 1.0 - 1.0 / n_classes:
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble can not be fit.')
            return (None, None, None)
        estimator_weight = self.learning_rate * (np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0))
        if not iboost == self.n_estimators - 1:
            sample_weight = np.exp(np.log(sample_weight) + estimator_weight * incorrect * (sample_weight > 0))
        return (sample_weight, estimator_weight, estimator_error)

    def predict(self, X):
        if False:
            while True:
                i = 10
        'Predict classes for X.\n\n        The predicted class of an input sample is computed as the weighted mean\n        prediction of the classifiers in the ensemble.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples. Sparse matrix can be CSC, CSR, COO,\n            DOK, or LIL. COO, DOK, and LIL are converted to CSR.\n\n        Returns\n        -------\n        y : ndarray of shape (n_samples,)\n            The predicted classes.\n        '
        pred = self.decision_function(X)
        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def staged_predict(self, X):
        if False:
            return 10
        'Return staged predictions for X.\n\n        The predicted class of an input sample is computed as the weighted mean\n        prediction of the classifiers in the ensemble.\n\n        This generator method yields the ensemble prediction after each\n        iteration of boosting and therefore allows monitoring, such as to\n        determine the prediction on a test set after each boost.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The input samples. Sparse matrix can be CSC, CSR, COO,\n            DOK, or LIL. COO, DOK, and LIL are converted to CSR.\n\n        Yields\n        ------\n        y : generator of ndarray of shape (n_samples,)\n            The predicted classes.\n        '
        X = self._check_X(X)
        n_classes = self.n_classes_
        classes = self.classes_
        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))
        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(np.argmax(pred, axis=1), axis=0))

    def decision_function(self, X):
        if False:
            return 10
        'Compute the decision function of ``X``.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples. Sparse matrix can be CSC, CSR, COO,\n            DOK, or LIL. COO, DOK, and LIL are converted to CSR.\n\n        Returns\n        -------\n        score : ndarray of shape of (n_samples, k)\n            The decision function of the input samples. The order of\n            outputs is the same as that of the :term:`classes_` attribute.\n            Binary classification is a special cases with ``k == 1``,\n            otherwise ``k==n_classes``. For binary classification,\n            values closer to -1 or 1 mean more like the first or second\n            class in ``classes_``, respectively.\n        '
        check_is_fitted(self)
        X = self._check_X(X)
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        if self.algorithm == 'SAMME.R':
            pred = sum((_samme_proba(estimator, n_classes, X) for estimator in self.estimators_))
        else:
            pred = sum((np.where((estimator.predict(X) == classes).T, w, -1 / (n_classes - 1) * w) for (estimator, w) in zip(self.estimators_, self.estimator_weights_)))
        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    def staged_decision_function(self, X):
        if False:
            i = 10
            return i + 15
        'Compute decision function of ``X`` for each boosting iteration.\n\n        This method allows monitoring (i.e. determine error on testing set)\n        after each boosting iteration.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples. Sparse matrix can be CSC, CSR, COO,\n            DOK, or LIL. COO, DOK, and LIL are converted to CSR.\n\n        Yields\n        ------\n        score : generator of ndarray of shape (n_samples, k)\n            The decision function of the input samples. The order of\n            outputs is the same of that of the :term:`classes_` attribute.\n            Binary classification is a special cases with ``k == 1``,\n            otherwise ``k==n_classes``. For binary classification,\n            values closer to -1 or 1 mean more like the first or second\n            class in ``classes_``, respectively.\n        '
        check_is_fitted(self)
        X = self._check_X(X)
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.0
        for (weight, estimator) in zip(self.estimator_weights_, self.estimators_):
            norm += weight
            if self.algorithm == 'SAMME.R':
                current_pred = _samme_proba(estimator, n_classes, X)
            else:
                current_pred = np.where((estimator.predict(X) == classes).T, weight, -1 / (n_classes - 1) * weight)
            if pred is None:
                pred = current_pred
            else:
                pred += current_pred
            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield (pred / norm)

    @staticmethod
    def _compute_proba_from_decision(decision, n_classes):
        if False:
            for i in range(10):
                print('nop')
        'Compute probabilities from the decision function.\n\n        This is based eq. (15) of [1] where:\n            p(y=c|X) = exp((1 / K-1) f_c(X)) / sum_k(exp((1 / K-1) f_k(X)))\n                     = softmax((1 / K-1) * f(X))\n\n        References\n        ----------\n        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost",\n               2009.\n        '
        if n_classes == 2:
            decision = np.vstack([-decision, decision]).T / 2
        else:
            decision /= n_classes - 1
        return softmax(decision, copy=False)

    def predict_proba(self, X):
        if False:
            return 10
        'Predict class probabilities for X.\n\n        The predicted class probabilities of an input sample is computed as\n        the weighted mean predicted class probabilities of the classifiers\n        in the ensemble.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples. Sparse matrix can be CSC, CSR, COO,\n            DOK, or LIL. COO, DOK, and LIL are converted to CSR.\n\n        Returns\n        -------\n        p : ndarray of shape (n_samples, n_classes)\n            The class probabilities of the input samples. The order of\n            outputs is the same of that of the :term:`classes_` attribute.\n        '
        check_is_fitted(self)
        n_classes = self.n_classes_
        if n_classes == 1:
            return np.ones((_num_samples(X), 1))
        decision = self.decision_function(X)
        return self._compute_proba_from_decision(decision, n_classes)

    def staged_predict_proba(self, X):
        if False:
            while True:
                i = 10
        'Predict class probabilities for X.\n\n        The predicted class probabilities of an input sample is computed as\n        the weighted mean predicted class probabilities of the classifiers\n        in the ensemble.\n\n        This generator method yields the ensemble predicted class probabilities\n        after each iteration of boosting and therefore allows monitoring, such\n        as to determine the predicted class probabilities on a test set after\n        each boost.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples. Sparse matrix can be CSC, CSR, COO,\n            DOK, or LIL. COO, DOK, and LIL are converted to CSR.\n\n        Yields\n        ------\n        p : generator of ndarray of shape (n_samples,)\n            The class probabilities of the input samples. The order of\n            outputs is the same of that of the :term:`classes_` attribute.\n        '
        n_classes = self.n_classes_
        for decision in self.staged_decision_function(X):
            yield self._compute_proba_from_decision(decision, n_classes)

    def predict_log_proba(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Predict class log-probabilities for X.\n\n        The predicted class log-probabilities of an input sample is computed as\n        the weighted mean predicted class log-probabilities of the classifiers\n        in the ensemble.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples. Sparse matrix can be CSC, CSR, COO,\n            DOK, or LIL. COO, DOK, and LIL are converted to CSR.\n\n        Returns\n        -------\n        p : ndarray of shape (n_samples, n_classes)\n            The class probabilities of the input samples. The order of\n            outputs is the same of that of the :term:`classes_` attribute.\n        '
        return np.log(self.predict_proba(X))

class AdaBoostRegressor(_RoutingNotSupportedMixin, RegressorMixin, BaseWeightBoosting):
    """An AdaBoost regressor.

    An AdaBoost [1] regressor is a meta-estimator that begins by fitting a
    regressor on the original dataset and then fits additional copies of the
    regressor on the same dataset but where the weights of instances are
    adjusted according to the error of the current prediction. As such,
    subsequent regressors focus more on difficult cases.

    This class implements the algorithm known as AdaBoost.R2 [2].

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        If ``None``, then the base estimator is
        :class:`~sklearn.tree.DecisionTreeRegressor` initialized with
        `max_depth=3`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
        Weight applied to each regressor at each boosting iteration. A higher
        learning rate increases the contribution of each regressor. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`.

    loss : {'linear', 'square', 'exponential'}, default='linear'
        The loss function to use when updating the weights after each
        boosting iteration.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `estimator` at each
        boosting iteration.
        Thus, it is only used when `estimator` exposes a `random_state`.
        In addition, it controls the bootstrap of the weights used to train the
        `estimator` at each boosting iteration.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    base_estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        If ``None``, then the base estimator is
        :class:`~sklearn.tree.DecisionTreeRegressor` initialized with
        `max_depth=3`.

        .. deprecated:: 1.2
            `base_estimator` is deprecated and will be removed in 1.4.
            Use `estimator` instead.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. deprecated:: 1.2
            `base_estimator_` is deprecated and will be removed in 1.4.
            Use `estimator_` instead.

    estimators_ : list of regressors
        The collection of fitted sub-estimators.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Regression error for each estimator in the boosted ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances if supported by the
        ``estimator`` (when based on decision trees).

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdaBoostClassifier : An AdaBoost classifier.
    GradientBoostingRegressor : Gradient Boosting Classification Tree.
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    >>> regr.fit(X, y)
    AdaBoostRegressor(n_estimators=100, random_state=0)
    >>> regr.predict([[0, 0, 0, 0]])
    array([4.7972...])
    >>> regr.score(X, y)
    0.9771...
    """
    _parameter_constraints: dict = {**BaseWeightBoosting._parameter_constraints, 'loss': [StrOptions({'linear', 'square', 'exponential'})]}

    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None, base_estimator='deprecated'):
        if False:
            return 10
        super().__init__(estimator=estimator, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, base_estimator=base_estimator)
        self.loss = loss
        self.random_state = random_state

    def _validate_estimator(self):
        if False:
            return 10
        'Check the estimator and set the estimator_ attribute.'
        super()._validate_estimator(default=DecisionTreeRegressor(max_depth=3))

    def _boost(self, iboost, X, y, sample_weight, random_state):
        if False:
            while True:
                i = 10
        'Implement a single boost for regression\n\n        Perform a single boost according to the AdaBoost.R2 algorithm and\n        return the updated sample weights.\n\n        Parameters\n        ----------\n        iboost : int\n            The index of the current boost iteration.\n\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples.\n\n        y : array-like of shape (n_samples,)\n            The target values (class labels in classification, real numbers in\n            regression).\n\n        sample_weight : array-like of shape (n_samples,)\n            The current sample weights.\n\n        random_state : RandomState\n            The RandomState instance used if the base estimator accepts a\n            `random_state` attribute.\n            Controls also the bootstrap of the weights used to train the weak\n            learner.\n            replacement.\n\n        Returns\n        -------\n        sample_weight : array-like of shape (n_samples,) or None\n            The reweighted sample weights.\n            If None then boosting has terminated early.\n\n        estimator_weight : float\n            The weight for the current boost.\n            If None then boosting has terminated early.\n\n        estimator_error : float\n            The regression error for the current boost.\n            If None then boosting has terminated early.\n        '
        estimator = self._make_estimator(random_state=random_state)
        bootstrap_idx = random_state.choice(np.arange(_num_samples(X)), size=_num_samples(X), replace=True, p=sample_weight)
        X_ = _safe_indexing(X, bootstrap_idx)
        y_ = _safe_indexing(y, bootstrap_idx)
        estimator.fit(X_, y_)
        y_predict = estimator.predict(X)
        error_vect = np.abs(y_predict - y)
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]
        error_max = masked_error_vector.max()
        if error_max != 0:
            masked_error_vector /= error_max
        if self.loss == 'square':
            masked_error_vector **= 2
        elif self.loss == 'exponential':
            masked_error_vector = 1.0 - np.exp(-masked_error_vector)
        estimator_error = (masked_sample_weight * masked_error_vector).sum()
        if estimator_error <= 0:
            return (sample_weight, 1.0, 0.0)
        elif estimator_error >= 0.5:
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return (None, None, None)
        beta = estimator_error / (1.0 - estimator_error)
        estimator_weight = self.learning_rate * np.log(1.0 / beta)
        if not iboost == self.n_estimators - 1:
            sample_weight[sample_mask] *= np.power(beta, (1.0 - masked_error_vector) * self.learning_rate)
        return (sample_weight, estimator_weight, estimator_error)

    def _get_median_predict(self, X, limit):
        if False:
            print('Hello World!')
        predictions = np.array([est.predict(X) for est in self.estimators_[:limit]]).T
        sorted_idx = np.argsort(predictions, axis=1)
        weight_cdf = stable_cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)
        median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]
        return predictions[np.arange(_num_samples(X)), median_estimators]

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Predict regression value for X.\n\n        The predicted regression value of an input sample is computed\n        as the weighted median prediction of the regressors in the ensemble.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples. Sparse matrix can be CSC, CSR, COO,\n            DOK, or LIL. COO, DOK, and LIL are converted to CSR.\n\n        Returns\n        -------\n        y : ndarray of shape (n_samples,)\n            The predicted regression values.\n        '
        check_is_fitted(self)
        X = self._check_X(X)
        return self._get_median_predict(X, len(self.estimators_))

    def staged_predict(self, X):
        if False:
            i = 10
            return i + 15
        'Return staged predictions for X.\n\n        The predicted regression value of an input sample is computed\n        as the weighted median prediction of the regressors in the ensemble.\n\n        This generator method yields the ensemble prediction after each\n        iteration of boosting and therefore allows monitoring, such as to\n        determine the prediction on a test set after each boost.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The training input samples.\n\n        Yields\n        ------\n        y : generator of ndarray of shape (n_samples,)\n            The predicted regression values.\n        '
        check_is_fitted(self)
        X = self._check_X(X)
        for (i, _) in enumerate(self.estimators_, 1):
            yield self._get_median_predict(X, limit=i)