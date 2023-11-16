"""Gradient Boosted Regression Trees.

This module contains methods for fitting gradient boosted regression trees for
both classification and regression.

The module structure is the following:

- The ``BaseGradientBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ in the concrete ``LossFunction`` used.

- ``GradientBoostingClassifier`` implements gradient boosting for
  classification problems.

- ``GradientBoostingRegressor`` implements gradient boosting for
  regression problems.
"""
import math
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from time import time
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, issparse
from .._loss.loss import _LOSSES, AbsoluteError, ExponentialLoss, HalfBinomialLoss, HalfMultinomialLoss, HalfSquaredError, HuberLoss, PinballLoss
from ..base import ClassifierMixin, RegressorMixin, _fit_context, is_classifier
from ..dummy import DummyClassifier, DummyRegressor
from ..exceptions import NotFittedError
from ..model_selection import train_test_split
from ..preprocessing import LabelEncoder
from ..tree import DecisionTreeRegressor
from ..tree._tree import DOUBLE, DTYPE, TREE_LEAF
from ..utils import check_array, check_random_state, column_or_1d
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.stats import _weighted_percentile
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import BaseEnsemble
from ._gradient_boosting import _random_sample_mask, predict_stage, predict_stages
_LOSSES = _LOSSES.copy()
_LOSSES.update({'quantile': PinballLoss, 'huber': HuberLoss})

def _safe_divide(numerator, denominator):
    if False:
        while True:
            i = 10
    'Prevents overflow and division by zero.'
    try:
        result = float(numerator) / float(denominator)
        if math.isinf(result):
            warnings.warn('overflow encountered in _safe_divide', RuntimeWarning)
        return result
    except ZeroDivisionError:
        warnings.warn('divide by zero encountered in _safe_divide', RuntimeWarning)
        return 0.0

def _init_raw_predictions(X, estimator, loss, use_predict_proba):
    if False:
        print('Hello World!')
    'Return the initial raw predictions.\n\n    Parameters\n    ----------\n    X : ndarray of shape (n_samples, n_features)\n        The data array.\n    estimator : object\n        The estimator to use to compute the predictions.\n    loss : BaseLoss\n        An instance of a loss function class.\n    use_predict_proba : bool\n        Whether estimator.predict_proba is used instead of estimator.predict.\n\n    Returns\n    -------\n    raw_predictions : ndarray of shape (n_samples, K)\n        The initial raw predictions. K is equal to 1 for binary\n        classification and regression, and equal to the number of classes\n        for multiclass classification. ``raw_predictions`` is casted\n        into float64.\n    '
    if use_predict_proba:
        predictions = estimator.predict_proba(X)
        if not loss.is_multiclass:
            predictions = predictions[:, 1]
        eps = np.finfo(np.float32).eps
        predictions = np.clip(predictions, eps, 1 - eps, dtype=np.float64)
    else:
        predictions = estimator.predict(X).astype(np.float64)
    if predictions.ndim == 1:
        return loss.link.link(predictions).reshape(-1, 1)
    else:
        return loss.link.link(predictions)

def _update_terminal_regions(loss, tree, X, y, neg_gradient, raw_prediction, sample_weight, sample_mask, learning_rate=0.1, k=0):
    if False:
        while True:
            i = 10
    'Update the leaf values to be predicted by the tree and raw_prediction.\n\n    The current raw predictions of the model (of this stage) are updated.\n\n    Additionally, the terminal regions (=leaves) of the given tree are updated as well.\n    This corresponds to the line search step in "Greedy Function Approximation" by\n    Friedman, Algorithm 1 step 5.\n\n    Update equals:\n        argmin_{x} loss(y_true, raw_prediction_old + x * tree.value)\n\n    For non-trivial cases like the Binomial loss, the update has no closed formula and\n    is an approximation, again, see the Friedman paper.\n\n    Also note that the update formula for the SquaredError is the identity. Therefore,\n    in this case, the leaf values don\'t need an update and only the raw_predictions are\n    updated (with the learning rate included).\n\n    Parameters\n    ----------\n    loss : BaseLoss\n    tree : tree.Tree\n        The tree object.\n    X : ndarray of shape (n_samples, n_features)\n        The data array.\n    y : ndarray of shape (n_samples,)\n        The target labels.\n    neg_gradient : ndarray of shape (n_samples,)\n        The negative gradient.\n    raw_prediction : ndarray of shape (n_samples, n_trees_per_iteration)\n        The raw predictions (i.e. values from the tree leaves) of the\n        tree ensemble at iteration ``i - 1``.\n    sample_weight : ndarray of shape (n_samples,)\n        The weight of each sample.\n    sample_mask : ndarray of shape (n_samples,)\n        The sample mask to be used.\n    learning_rate : float, default=0.1\n        Learning rate shrinks the contribution of each tree by\n         ``learning_rate``.\n    k : int, default=0\n        The index of the estimator being updated.\n    '
    terminal_regions = tree.apply(X)
    if not isinstance(loss, HalfSquaredError):
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1
        if isinstance(loss, HalfBinomialLoss):

            def compute_update(y_, indices, neg_gradient, raw_prediction, k):
                if False:
                    print('Hello World!')
                neg_g = neg_gradient.take(indices, axis=0)
                prob = y_ - neg_g
                numerator = np.average(neg_g, weights=sw)
                denominator = np.average(prob * (1 - prob), weights=sw)
                return _safe_divide(numerator, denominator)
        elif isinstance(loss, HalfMultinomialLoss):

            def compute_update(y_, indices, neg_gradient, raw_prediction, k):
                if False:
                    i = 10
                    return i + 15
                neg_g = neg_gradient.take(indices, axis=0)
                prob = y_ - neg_g
                K = loss.n_classes
                numerator = np.average(neg_g, weights=sw)
                numerator *= (K - 1) / K
                denominator = np.average(prob * (1 - prob), weights=sw)
                return _safe_divide(numerator, denominator)
        elif isinstance(loss, ExponentialLoss):

            def compute_update(y_, indices, neg_gradient, raw_prediction, k):
                if False:
                    print('Hello World!')
                neg_g = neg_gradient.take(indices, axis=0)
                numerator = np.average(neg_g, weights=sw)
                hessian = neg_g.copy()
                hessian[y_ == 0] *= -1
                denominator = np.average(hessian, weights=sw)
                return _safe_divide(numerator, denominator)
        else:

            def compute_update(y_, indices, neg_gradient, raw_prediction, k):
                if False:
                    for i in range(10):
                        print('nop')
                return loss.fit_intercept_only(y_true=y_ - raw_prediction[indices, k], sample_weight=sw)
        for leaf in np.nonzero(tree.children_left == TREE_LEAF)[0]:
            indices = np.nonzero(masked_terminal_regions == leaf)[0]
            y_ = y.take(indices, axis=0)
            sw = None if sample_weight is None else sample_weight[indices]
            update = compute_update(y_, indices, neg_gradient, raw_prediction, k)
            tree.value[leaf, 0, 0] = update
    raw_prediction[:, k] += learning_rate * tree.value[:, 0, 0].take(terminal_regions, axis=0)

def set_huber_delta(loss, y_true, raw_prediction, sample_weight=None):
    if False:
        i = 10
        return i + 15
    'Calculate and set self.closs.delta based on self.quantile.'
    abserr = np.abs(y_true - raw_prediction.squeeze())
    delta = _weighted_percentile(abserr, sample_weight, 100 * loss.quantile)
    loss.closs.delta = float(delta)

class VerboseReporter:
    """Reports verbose output to stdout.

    Parameters
    ----------
    verbose : int
        Verbosity level. If ``verbose==1`` output is printed once in a while
        (when iteration mod verbose_mod is zero).; if larger than 1 then output
        is printed for each update.
    """

    def __init__(self, verbose):
        if False:
            i = 10
            return i + 15
        self.verbose = verbose

    def init(self, est, begin_at_stage=0):
        if False:
            return 10
        'Initialize reporter\n\n        Parameters\n        ----------\n        est : Estimator\n            The estimator\n\n        begin_at_stage : int, default=0\n            stage at which to begin reporting\n        '
        header_fields = ['Iter', 'Train Loss']
        verbose_fmt = ['{iter:>10d}', '{train_score:>16.4f}']
        if est.subsample < 1:
            header_fields.append('OOB Improve')
            verbose_fmt.append('{oob_impr:>16.4f}')
        header_fields.append('Remaining Time')
        verbose_fmt.append('{remaining_time:>16s}')
        print(('%10s ' + '%16s ' * (len(header_fields) - 1)) % tuple(header_fields))
        self.verbose_fmt = ' '.join(verbose_fmt)
        self.verbose_mod = 1
        self.start_time = time()
        self.begin_at_stage = begin_at_stage

    def update(self, j, est):
        if False:
            for i in range(10):
                print('nop')
        'Update reporter with new iteration.\n\n        Parameters\n        ----------\n        j : int\n            The new iteration.\n        est : Estimator\n            The estimator.\n        '
        do_oob = est.subsample < 1
        i = j - self.begin_at_stage
        if (i + 1) % self.verbose_mod == 0:
            oob_impr = est.oob_improvement_[j] if do_oob else 0
            remaining_time = (est.n_estimators - (j + 1)) * (time() - self.start_time) / float(i + 1)
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)
            print(self.verbose_fmt.format(iter=j + 1, train_score=est.train_score_[j], oob_impr=oob_impr, remaining_time=remaining_time))
            if self.verbose == 1 and (i + 1) // (self.verbose_mod * 10) > 0:
                self.verbose_mod *= 10

class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Abstract base class for Gradient Boosting."""
    _parameter_constraints: dict = {**DecisionTreeRegressor._parameter_constraints, 'learning_rate': [Interval(Real, 0.0, None, closed='left')], 'n_estimators': [Interval(Integral, 1, None, closed='left')], 'criterion': [StrOptions({'friedman_mse', 'squared_error'})], 'subsample': [Interval(Real, 0.0, 1.0, closed='right')], 'verbose': ['verbose'], 'warm_start': ['boolean'], 'validation_fraction': [Interval(Real, 0.0, 1.0, closed='neither')], 'n_iter_no_change': [Interval(Integral, 1, None, closed='left'), None], 'tol': [Interval(Real, 0.0, None, closed='left')]}
    _parameter_constraints.pop('splitter')
    _parameter_constraints.pop('monotonic_cst')

    @abstractmethod
    def __init__(self, *, loss, learning_rate, n_estimators, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth, min_impurity_decrease, init, subsample, max_features, ccp_alpha, random_state, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001):
        if False:
            for i in range(10):
                print('nop')
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

    @abstractmethod
    def _encode_y(self, y=None, sample_weight=None):
        if False:
            while True:
                i = 10
        'Called by fit to validate and encode y.'

    @abstractmethod
    def _get_loss(self, sample_weight):
        if False:
            while True:
                i = 10
        'Get loss object from sklearn._loss.loss.'

    def _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask, random_state, X_csc=None, X_csr=None):
        if False:
            return 10
        'Fit another stage of ``n_trees_per_iteration_`` trees.'
        original_y = y
        if isinstance(self._loss, HuberLoss):
            set_huber_delta(loss=self._loss, y_true=y, raw_prediction=raw_predictions, sample_weight=sample_weight)
        neg_gradient = -self._loss.gradient(y_true=y, raw_prediction=raw_predictions, sample_weight=None)
        if neg_gradient.ndim == 1:
            neg_g_view = neg_gradient.reshape((-1, 1))
        else:
            neg_g_view = neg_gradient
        for k in range(self.n_trees_per_iteration_):
            if self._loss.is_multiclass:
                y = np.array(original_y == k, dtype=np.float64)
            tree = DecisionTreeRegressor(criterion=self.criterion, splitter='best', max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, min_weight_fraction_leaf=self.min_weight_fraction_leaf, min_impurity_decrease=self.min_impurity_decrease, max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes, random_state=random_state, ccp_alpha=self.ccp_alpha)
            if self.subsample < 1.0:
                sample_weight = sample_weight * sample_mask.astype(np.float64)
            X = X_csc if X_csc is not None else X
            tree.fit(X, neg_g_view[:, k], sample_weight=sample_weight, check_input=False)
            X_for_tree_update = X_csr if X_csr is not None else X
            _update_terminal_regions(self._loss, tree.tree_, X_for_tree_update, y, neg_g_view[:, k], raw_predictions, sample_weight, sample_mask, learning_rate=self.learning_rate, k=k)
            self.estimators_[i, k] = tree
        return raw_predictions

    def _set_max_features(self):
        if False:
            print('Hello World!')
        'Set self.max_features_.'
        if isinstance(self.max_features, str):
            if self.max_features == 'auto':
                if is_classifier(self):
                    max_features = max(1, int(np.sqrt(self.n_features_in_)))
                else:
                    max_features = self.n_features_in_
            elif self.max_features == 'sqrt':
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            else:
                max_features = max(1, int(np.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, Integral):
            max_features = self.max_features
        else:
            max_features = max(1, int(self.max_features * self.n_features_in_))
        self.max_features_ = max_features

    def _init_state(self):
        if False:
            print('Hello World!')
        'Initialize model state and allocate model state data structures.'
        self.init_ = self.init
        if self.init_ is None:
            if is_classifier(self):
                self.init_ = DummyClassifier(strategy='prior')
            elif isinstance(self._loss, (AbsoluteError, HuberLoss)):
                self.init_ = DummyRegressor(strategy='quantile', quantile=0.5)
            elif isinstance(self._loss, PinballLoss):
                self.init_ = DummyRegressor(strategy='quantile', quantile=self.alpha)
            else:
                self.init_ = DummyRegressor(strategy='mean')
        self.estimators_ = np.empty((self.n_estimators, self.n_trees_per_iteration_), dtype=object)
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros(self.n_estimators, dtype=np.float64)
            self.oob_scores_ = np.zeros(self.n_estimators, dtype=np.float64)
            self.oob_score_ = np.nan

    def _clear_state(self):
        if False:
            print('Hello World!')
        'Clear the state of the gradient boosting model.'
        if hasattr(self, 'estimators_'):
            self.estimators_ = np.empty((0, 0), dtype=object)
        if hasattr(self, 'train_score_'):
            del self.train_score_
        if hasattr(self, 'oob_improvement_'):
            del self.oob_improvement_
        if hasattr(self, 'oob_scores_'):
            del self.oob_scores_
        if hasattr(self, 'oob_score_'):
            del self.oob_score_
        if hasattr(self, 'init_'):
            del self.init_
        if hasattr(self, '_rng'):
            del self._rng

    def _resize_state(self):
        if False:
            print('Hello World!')
        'Add additional ``n_estimators`` entries to all attributes.'
        total_n_estimators = self.n_estimators
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError('resize with smaller n_estimators %d < %d' % (total_n_estimators, self.estimators_[0]))
        self.estimators_ = np.resize(self.estimators_, (total_n_estimators, self.n_trees_per_iteration_))
        self.train_score_ = np.resize(self.train_score_, total_n_estimators)
        if self.subsample < 1 or hasattr(self, 'oob_improvement_'):
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = np.resize(self.oob_improvement_, total_n_estimators)
                self.oob_scores_ = np.resize(self.oob_scores_, total_n_estimators)
                self.oob_score_ = np.nan
            else:
                self.oob_improvement_ = np.zeros((total_n_estimators,), dtype=np.float64)
                self.oob_scores_ = np.zeros((total_n_estimators,), dtype=np.float64)
                self.oob_score_ = np.nan

    def _is_fitted(self):
        if False:
            while True:
                i = 10
        return len(getattr(self, 'estimators_', [])) > 0

    def _check_initialized(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that the estimator is initialized, raising an error if not.'
        check_is_fitted(self)

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, sample_weight=None, monitor=None):
        if False:
            while True:
                i = 10
        'Fit the gradient boosting model.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        y : array-like of shape (n_samples,)\n            Target values (strings or integers in classification, real numbers\n            in regression)\n            For classification, labels must correspond to classes.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights. If None, then samples are equally weighted. Splits\n            that would create child nodes with net zero or negative weight are\n            ignored while searching for a split in each node. In the case of\n            classification, splits are also ignored if they would result in any\n            single class carrying a negative weight in either child node.\n\n        monitor : callable, default=None\n            The monitor is called after each iteration with the current\n            iteration, a reference to the estimator and the local variables of\n            ``_fit_stages`` as keyword arguments ``callable(i, self,\n            locals())``. If the callable returns ``True`` the fitting procedure\n            is stopped. The monitor can be used for various things such as\n            computing held-out estimates, early stopping, model introspect, and\n            snapshoting.\n\n        Returns\n        -------\n        self : object\n            Fitted estimator.\n        '
        if not self.warm_start:
            self._clear_state()
        (X, y) = self._validate_data(X, y, accept_sparse=['csr', 'csc', 'coo'], dtype=DTYPE, multi_output=True)
        sample_weight_is_none = sample_weight is None
        sample_weight = _check_sample_weight(sample_weight, X)
        if sample_weight_is_none:
            y = self._encode_y(y=y, sample_weight=None)
        else:
            y = self._encode_y(y=y, sample_weight=sample_weight)
        y = column_or_1d(y, warn=True)
        self._set_max_features()
        self._loss = self._get_loss(sample_weight=sample_weight)
        if self.n_iter_no_change is not None:
            stratify = y if is_classifier(self) else None
            (X_train, X_val, y_train, y_val, sample_weight_train, sample_weight_val) = train_test_split(X, y, sample_weight, random_state=self.random_state, test_size=self.validation_fraction, stratify=stratify)
            if is_classifier(self):
                if self.n_classes_ != np.unique(y_train).shape[0]:
                    raise ValueError('The training data after the early stopping split is missing some classes. Try using another random seed.')
        else:
            (X_train, y_train, sample_weight_train) = (X, y, sample_weight)
            X_val = y_val = sample_weight_val = None
        n_samples = X_train.shape[0]
        if not self._is_fitted():
            self._init_state()
            if self.init_ == 'zero':
                raw_predictions = np.zeros(shape=(n_samples, self.n_trees_per_iteration_), dtype=np.float64)
            else:
                if sample_weight_is_none:
                    self.init_.fit(X_train, y_train)
                else:
                    msg = 'The initial estimator {} does not support sample weights.'.format(self.init_.__class__.__name__)
                    try:
                        self.init_.fit(X_train, y_train, sample_weight=sample_weight_train)
                    except TypeError as e:
                        if "unexpected keyword argument 'sample_weight'" in str(e):
                            raise ValueError(msg) from e
                        else:
                            raise
                    except ValueError as e:
                        if 'pass parameters to specific steps of your pipeline using the stepname__parameter' in str(e):
                            raise ValueError(msg) from e
                        else:
                            raise
                raw_predictions = _init_raw_predictions(X_train, self.init_, self._loss, is_classifier(self))
            begin_at_stage = 0
            self._rng = check_random_state(self.random_state)
        else:
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to estimators_.shape[0]=%d when warm_start==True' % (self.n_estimators, self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            X_train = check_array(X_train, dtype=DTYPE, order='C', accept_sparse='csr', force_all_finite=False)
            raw_predictions = self._raw_predict(X_train)
            self._resize_state()
        n_stages = self._fit_stages(X_train, y_train, raw_predictions, sample_weight_train, self._rng, X_val, y_val, sample_weight_val, begin_at_stage, monitor)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]
                self.oob_scores_ = self.oob_scores_[:n_stages]
                self.oob_score_ = self.oob_scores_[-1]
        self.n_estimators_ = n_stages
        return self

    def _fit_stages(self, X, y, raw_predictions, sample_weight, random_state, X_val, y_val, sample_weight_val, begin_at_stage=0, monitor=None):
        if False:
            for i in range(10):
                print('nop')
        'Iteratively fits the stages.\n\n        For each stage it computes the progress (OOB, train score)\n        and delegates to ``_fit_stage``.\n        Returns the number of stages fit; might differ from ``n_estimators``\n        due to early stopping.\n        '
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples,), dtype=bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)
        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None
        if self.n_iter_no_change is not None:
            loss_history = np.full(self.n_iter_no_change, np.inf)
            y_val_pred_iter = self._staged_raw_predict(X_val, check_input=False)
        if isinstance(self._loss, (HalfSquaredError, HalfBinomialLoss)):
            factor = 2
        else:
            factor = 1
        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag, random_state)
                y_oob_masked = y[~sample_mask]
                sample_weight_oob_masked = sample_weight[~sample_mask]
                if i == 0:
                    initial_loss = factor * self._loss(y_true=y_oob_masked, raw_prediction=raw_predictions[~sample_mask], sample_weight=sample_weight_oob_masked)
            raw_predictions = self._fit_stage(i, X, y, raw_predictions, sample_weight, sample_mask, random_state, X_csc=X_csc, X_csr=X_csr)
            if do_oob:
                self.train_score_[i] = factor * self._loss(y_true=y[sample_mask], raw_prediction=raw_predictions[sample_mask], sample_weight=sample_weight[sample_mask])
                self.oob_scores_[i] = factor * self._loss(y_true=y_oob_masked, raw_prediction=raw_predictions[~sample_mask], sample_weight=sample_weight_oob_masked)
                previous_loss = initial_loss if i == 0 else self.oob_scores_[i - 1]
                self.oob_improvement_[i] = previous_loss - self.oob_scores_[i]
                self.oob_score_ = self.oob_scores_[-1]
            else:
                self.train_score_[i] = factor * self._loss(y_true=y, raw_prediction=raw_predictions, sample_weight=sample_weight)
            if self.verbose > 0:
                verbose_reporter.update(i, self)
            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break
            if self.n_iter_no_change is not None:
                validation_loss = factor * self._loss(y_val, next(y_val_pred_iter), sample_weight_val)
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break
        return i + 1

    def _make_estimator(self, append=True):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def _raw_predict_init(self, X):
        if False:
            return 10
        'Check input and compute raw predictions of the init estimator.'
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if self.init_ == 'zero':
            raw_predictions = np.zeros(shape=(X.shape[0], self.n_trees_per_iteration_), dtype=np.float64)
        else:
            raw_predictions = _init_raw_predictions(X, self.init_, self._loss, is_classifier(self))
        return raw_predictions

    def _raw_predict(self, X):
        if False:
            return 10
        'Return the sum of the trees raw predictions (+ init estimator).'
        check_is_fitted(self)
        raw_predictions = self._raw_predict_init(X)
        predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)
        return raw_predictions

    def _staged_raw_predict(self, X, check_input=True):
        if False:
            print('Hello World!')
        'Compute raw predictions of ``X`` for each iteration.\n\n        This method allows monitoring (i.e. determine error on testing set)\n        after each stage.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        check_input : bool, default=True\n            If False, the input arrays X will not be checked.\n\n        Returns\n        -------\n        raw_predictions : generator of ndarray of shape (n_samples, k)\n            The raw predictions of the input samples. The order of the\n            classes corresponds to that in the attribute :term:`classes_`.\n            Regression and binary classification are special cases with\n            ``k == 1``, otherwise ``k==n_classes``.\n        '
        if check_input:
            X = self._validate_data(X, dtype=DTYPE, order='C', accept_sparse='csr', reset=False)
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            predict_stage(self.estimators_, i, X, self.learning_rate, raw_predictions)
            yield raw_predictions.copy()

    @property
    def feature_importances_(self):
        if False:
            print('Hello World!')
        'The impurity-based feature importances.\n\n        The higher, the more important the feature.\n        The importance of a feature is computed as the (normalized)\n        total reduction of the criterion brought by that feature.  It is also\n        known as the Gini importance.\n\n        Warning: impurity-based feature importances can be misleading for\n        high cardinality features (many unique values). See\n        :func:`sklearn.inspection.permutation_importance` as an alternative.\n\n        Returns\n        -------\n        feature_importances_ : ndarray of shape (n_features,)\n            The values of this array sum to 1, unless all trees are single node\n            trees consisting of only the root node, in which case it will be an\n            array of zeros.\n        '
        self._check_initialized()
        relevant_trees = [tree for stage in self.estimators_ for tree in stage if tree.tree_.node_count > 1]
        if not relevant_trees:
            return np.zeros(shape=self.n_features_in_, dtype=np.float64)
        relevant_feature_importances = [tree.tree_.compute_feature_importances(normalize=False) for tree in relevant_trees]
        avg_feature_importances = np.mean(relevant_feature_importances, axis=0, dtype=np.float64)
        return avg_feature_importances / np.sum(avg_feature_importances)

    def _compute_partial_dependence_recursion(self, grid, target_features):
        if False:
            while True:
                i = 10
        'Fast partial dependence computation.\n\n        Parameters\n        ----------\n        grid : ndarray of shape (n_samples, n_target_features)\n            The grid points on which the partial dependence should be\n            evaluated.\n        target_features : ndarray of shape (n_target_features,)\n            The set of target features for which the partial dependence\n            should be evaluated.\n\n        Returns\n        -------\n        averaged_predictions : ndarray of shape                 (n_trees_per_iteration_, n_samples)\n            The value of the partial dependence function on each grid point.\n        '
        if self.init is not None:
            warnings.warn('Using recursion method with a non-constant init predictor will lead to incorrect partial dependence values. Got init=%s.' % self.init, UserWarning)
        grid = np.asarray(grid, dtype=DTYPE, order='C')
        (n_estimators, n_trees_per_stage) = self.estimators_.shape
        averaged_predictions = np.zeros((n_trees_per_stage, grid.shape[0]), dtype=np.float64, order='C')
        for stage in range(n_estimators):
            for k in range(n_trees_per_stage):
                tree = self.estimators_[stage, k].tree_
                tree.compute_partial_dependence(grid, target_features, averaged_predictions[k])
        averaged_predictions *= self.learning_rate
        return averaged_predictions

    def apply(self, X):
        if False:
            print('Hello World!')
        'Apply trees in the ensemble to X, return leaf indices.\n\n        .. versionadded:: 0.17\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, its dtype will be converted to\n            ``dtype=np.float32``. If a sparse matrix is provided, it will\n            be converted to a sparse ``csr_matrix``.\n\n        Returns\n        -------\n        X_leaves : array-like of shape (n_samples, n_estimators, n_classes)\n            For each datapoint x in X and for each tree in the ensemble,\n            return the index of the leaf x ends up in each estimator.\n            In the case of binary classification n_classes is 1.\n        '
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        (n_estimators, n_classes) = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))
        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)
        return leaves

class GradientBoostingClassifier(ClassifierMixin, BaseGradientBoosting):
    """Gradient Boosting for classification.

    This algorithm builds an additive model in a forward stage-wise fashion; it
    allows for the optimization of arbitrary differentiable loss functions. In
    each stage ``n_classes_`` regression trees are fit on the negative gradient
    of the loss function, e.g. binary or multiclass log loss. Binary
    classification is a special case where only a single regression tree is
    induced.

    :class:`sklearn.ensemble.HistGradientBoostingClassifier` is a much faster
    variant of this algorithm for intermediate datasets (`n_samples >= 10_000`).

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'log_loss', 'exponential'}, default='log_loss'
        The loss function to be optimized. 'log_loss' refers to binomial and
        multinomial deviance, the same as used in logistic regression.
        It is a good choice for classification with probabilistic outputs.
        For loss 'exponential', gradient boosting recovers the AdaBoost algorithm.

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        Values must be in the range `[0.0, inf)`.

    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range `[1, inf)`.

    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
        Values must be in the range `(0.0, 1.0]`.

    criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria are
        'friedman_mse' for the mean squared error with improvement score by
        Friedman, 'squared_error' for mean squared error. The default value of
        'friedman_mse' is generally the best as it can provide a better
        approximation in some cases.

        .. versionadded:: 0.18

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, values must be in the range `[2, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`
          will be `ceil(min_samples_split * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`
          will be `ceil(min_samples_leaf * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
        Values must be in the range `[0.0, 0.5]`.

    max_depth : int or None, default=3
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        If int, values must be in the range `[1, inf)`.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        Values must be in the range `[0.0, inf)`.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    init : estimator or 'zero', default=None
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :term:`fit` and :term:`predict_proba`. If
        'zero', the initial raw predictions are set to zero. By default, a
        ``DummyEstimator`` predicting the classes priors is used.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
        In addition, it controls the random permutation of the features at
        each split (see Notes for more details).
        It also controls the random splitting of the training data to obtain a
        validation set if `n_iter_no_change` is not None.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    max_features : {'sqrt', 'log2'}, int or float, default=None
        The number of features to consider when looking for the best split:

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and the features
          considered at each split will be `max(1, int(max_features * n_features_in_))`.
        - If 'sqrt', then `max_features=sqrt(n_features)`.
        - If 'log2', then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
        Values must be in the range `[0, inf)`.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        Values must be in the range `[2, inf)`.
        If `None`, then unlimited number of leaf nodes.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Values must be in the range `(0.0, 1.0)`.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations. The split is stratified.
        Values must be in the range `[1, inf)`.
        See
        :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_early_stopping.py`.

        .. versionadded:: 0.20

    tol : float, default=1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        Values must be in the range `[0.0, inf)`.

        .. versionadded:: 0.20

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.
        Values must be in the range `[0.0, inf)`.
        See :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

        .. versionadded:: 0.20

    n_trees_per_iteration_ : int
        The number of trees that are built at each iteration. For binary classifiers,
        this is always 1.

        .. versionadded:: 1.4.0

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_improvement_ : ndarray of shape (n_estimators,)
        The improvement in loss on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``.

    oob_scores_ : ndarray of shape (n_estimators,)
        The full history of the loss values on the out-of-bag
        samples. Only available if `subsample < 1.0`.

        .. versionadded:: 1.3

    oob_score_ : float
        The last value of the loss on the out-of-bag samples. It is
        the same as `oob_scores_[-1]`. Only available if `subsample < 1.0`.

        .. versionadded:: 1.3

    train_score_ : ndarray of shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the loss of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the loss on the training data.

    init_ : estimator
        The estimator that provides the initial predictions. Set via the ``init``
        argument.

    estimators_ : ndarray of DecisionTreeRegressor of             shape (n_estimators, ``n_trees_per_iteration_``)
        The collection of fitted sub-estimators. ``n_trees_per_iteration_`` is 1 for
        binary classification, otherwise ``n_classes``.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_classes_ : int
        The number of classes.

    max_features_ : int
        The inferred value of max_features.

    See Also
    --------
    HistGradientBoostingClassifier : Histogram-based Gradient Boosting
        Classification Tree.
    sklearn.tree.DecisionTreeClassifier : A decision tree classifier.
    RandomForestClassifier : A meta-estimator that fits a number of decision
        tree classifiers on various sub-samples of the dataset and uses
        averaging to improve the predictive accuracy and control over-fitting.
    AdaBoostClassifier : A meta-estimator that begins by fitting a classifier
        on the original dataset and then fits additional copies of the
        classifier on the same dataset where the weights of incorrectly
        classified instances are adjusted such that subsequent classifiers
        focus more on difficult cases.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.

    Examples
    --------
    The following example shows how to fit a gradient boosting classifier with
    100 decision stumps as weak learners.

    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> X_train, X_test = X[:2000], X[2000:]
    >>> y_train, y_test = y[:2000], y[2000:]

    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X_train, y_train)
    >>> clf.score(X_test, y_test)
    0.913...
    """
    _parameter_constraints: dict = {**BaseGradientBoosting._parameter_constraints, 'loss': [StrOptions({'log_loss', 'exponential'})], 'init': [StrOptions({'zero'}), None, HasMethods(['fit', 'predict_proba'])]}

    def __init__(self, *, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
        if False:
            while True:
                i = 10
        super().__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, init=init, subsample=subsample, max_features=max_features, random_state=random_state, verbose=verbose, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, warm_start=warm_start, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

    def _encode_y(self, y, sample_weight):
        if False:
            for i in range(10):
                print('nop')
        check_classification_targets(y)
        label_encoder = LabelEncoder()
        encoded_y_int = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        n_classes = self.classes_.shape[0]
        self.n_trees_per_iteration_ = 1 if n_classes <= 2 else n_classes
        encoded_y = encoded_y_int.astype(float, copy=False)
        self.n_classes_ = n_classes
        if sample_weight is None:
            n_trim_classes = n_classes
        else:
            n_trim_classes = np.count_nonzero(np.bincount(encoded_y_int, sample_weight))
        if n_trim_classes < 2:
            raise ValueError('y contains %d class after sample_weight trimmed classes with zero weights, while a minimum of 2 classes are required.' % n_trim_classes)
        return encoded_y

    def _get_loss(self, sample_weight):
        if False:
            print('Hello World!')
        if self.loss == 'log_loss':
            if self.n_classes_ == 2:
                return HalfBinomialLoss(sample_weight=sample_weight)
            else:
                return HalfMultinomialLoss(sample_weight=sample_weight, n_classes=self.n_classes_)
        elif self.loss == 'exponential':
            if self.n_classes_ > 2:
                raise ValueError(f"loss='{self.loss}' is only suitable for a binary classification problem, you have n_classes={self.n_classes_}. Please use loss='log_loss' instead.")
            else:
                return ExponentialLoss(sample_weight=sample_weight)

    def decision_function(self, X):
        if False:
            print('Hello World!')
        'Compute the decision function of ``X``.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        Returns\n        -------\n        score : ndarray of shape (n_samples, n_classes) or (n_samples,)\n            The decision function of the input samples, which corresponds to\n            the raw values predicted from the trees of the ensemble . The\n            order of the classes corresponds to that in the attribute\n            :term:`classes_`. Regression and binary classification produce an\n            array of shape (n_samples,).\n        '
        X = self._validate_data(X, dtype=DTYPE, order='C', accept_sparse='csr', reset=False)
        raw_predictions = self._raw_predict(X)
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions

    def staged_decision_function(self, X):
        if False:
            i = 10
            return i + 15
        'Compute decision function of ``X`` for each iteration.\n\n        This method allows monitoring (i.e. determine error on testing set)\n        after each stage.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        Yields\n        ------\n        score : generator of ndarray of shape (n_samples, k)\n            The decision function of the input samples, which corresponds to\n            the raw values predicted from the trees of the ensemble . The\n            classes corresponds to that in the attribute :term:`classes_`.\n            Regression and binary classification are special cases with\n            ``k == 1``, otherwise ``k==n_classes``.\n        '
        yield from self._staged_raw_predict(X)

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Predict class for X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        Returns\n        -------\n        y : ndarray of shape (n_samples,)\n            The predicted values.\n        '
        raw_predictions = self.decision_function(X)
        if raw_predictions.ndim == 1:
            encoded_classes = (raw_predictions >= 0).astype(int)
        else:
            encoded_classes = np.argmax(raw_predictions, axis=1)
        return self.classes_[encoded_classes]

    def staged_predict(self, X):
        if False:
            while True:
                i = 10
        'Predict class at each stage for X.\n\n        This method allows monitoring (i.e. determine error on testing set)\n        after each stage.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        Yields\n        ------\n        y : generator of ndarray of shape (n_samples,)\n            The predicted value of the input samples.\n        '
        if self.n_classes_ == 2:
            for raw_predictions in self._staged_raw_predict(X):
                encoded_classes = (raw_predictions.squeeze() >= 0).astype(int)
                yield self.classes_.take(encoded_classes, axis=0)
        else:
            for raw_predictions in self._staged_raw_predict(X):
                encoded_classes = np.argmax(raw_predictions, axis=1)
                yield self.classes_.take(encoded_classes, axis=0)

    def predict_proba(self, X):
        if False:
            print('Hello World!')
        'Predict class probabilities for X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        Returns\n        -------\n        p : ndarray of shape (n_samples, n_classes)\n            The class probabilities of the input samples. The order of the\n            classes corresponds to that in the attribute :term:`classes_`.\n\n        Raises\n        ------\n        AttributeError\n            If the ``loss`` does not support probabilities.\n        '
        raw_predictions = self.decision_function(X)
        return self._loss.predict_proba(raw_predictions)

    def predict_log_proba(self, X):
        if False:
            i = 10
            return i + 15
        'Predict class log-probabilities for X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        Returns\n        -------\n        p : ndarray of shape (n_samples, n_classes)\n            The class log-probabilities of the input samples. The order of the\n            classes corresponds to that in the attribute :term:`classes_`.\n\n        Raises\n        ------\n        AttributeError\n            If the ``loss`` does not support probabilities.\n        '
        proba = self.predict_proba(X)
        return np.log(proba)

    def staged_predict_proba(self, X):
        if False:
            while True:
                i = 10
        'Predict class probabilities at each stage for X.\n\n        This method allows monitoring (i.e. determine error on testing set)\n        after each stage.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        Yields\n        ------\n        y : generator of ndarray of shape (n_samples,)\n            The predicted value of the input samples.\n        '
        try:
            for raw_predictions in self._staged_raw_predict(X):
                yield self._loss.predict_proba(raw_predictions)
        except NotFittedError:
            raise
        except AttributeError as e:
            raise AttributeError('loss=%r does not support predict_proba' % self.loss) from e

class GradientBoostingRegressor(RegressorMixin, BaseGradientBoosting):
    """Gradient Boosting for regression.

    This estimator builds an additive model in a forward stage-wise fashion; it
    allows for the optimization of arbitrary differentiable loss functions. In
    each stage a regression tree is fit on the negative gradient of the given
    loss function.

    :class:`sklearn.ensemble.HistGradientBoostingRegressor` is a much faster
    variant of this algorithm for intermediate datasets (`n_samples >= 10_000`).

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'squared_error', 'absolute_error', 'huber', 'quantile'},             default='squared_error'
        Loss function to be optimized. 'squared_error' refers to the squared
        error for regression. 'absolute_error' refers to the absolute error of
        regression and is a robust loss function. 'huber' is a
        combination of the two. 'quantile' allows quantile regression (use
        `alpha` to specify the quantile).

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        Values must be in the range `[0.0, inf)`.

    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range `[1, inf)`.

    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
        Values must be in the range `(0.0, 1.0]`.

    criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria are
        "friedman_mse" for the mean squared error with improvement score by
        Friedman, "squared_error" for mean squared error. The default value of
        "friedman_mse" is generally the best as it can provide a better
        approximation in some cases.

        .. versionadded:: 0.18

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, values must be in the range `[2, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`
          will be `ceil(min_samples_split * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`
          will be `ceil(min_samples_leaf * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
        Values must be in the range `[0.0, 0.5]`.

    max_depth : int or None, default=3
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        If int, values must be in the range `[1, inf)`.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        Values must be in the range `[0.0, inf)`.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    init : estimator or 'zero', default=None
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :term:`fit` and :term:`predict`. If 'zero', the
        initial raw predictions are set to zero. By default a
        ``DummyEstimator`` is used, predicting either the average target value
        (for loss='squared_error'), or a quantile for the other losses.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
        In addition, it controls the random permutation of the features at
        each split (see Notes for more details).
        It also controls the random splitting of the training data to obtain a
        validation set if `n_iter_no_change` is not None.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    max_features : {'sqrt', 'log2'}, int or float, default=None
        The number of features to consider when looking for the best split:

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and the features
          considered at each split will be `max(1, int(max_features * n_features_in_))`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    alpha : float, default=0.9
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.
        Values must be in the range `(0.0, 1.0)`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
        Values must be in the range `[0, inf)`.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        Values must be in the range `[2, inf)`.
        If None, then unlimited number of leaf nodes.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Values must be in the range `(0.0, 1.0)`.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.
        Values must be in the range `[1, inf)`.
        See
        :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_early_stopping.py`.

        .. versionadded:: 0.20

    tol : float, default=1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        Values must be in the range `[0.0, inf)`.

        .. versionadded:: 0.20

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.
        Values must be in the range `[0.0, inf)`.
        See :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

    n_trees_per_iteration_ : int
        The number of trees that are built at each iteration. For regressors, this is
        always 1.

        .. versionadded:: 1.4.0

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_improvement_ : ndarray of shape (n_estimators,)
        The improvement in loss on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``.

    oob_scores_ : ndarray of shape (n_estimators,)
        The full history of the loss values on the out-of-bag
        samples. Only available if `subsample < 1.0`.

        .. versionadded:: 1.3

    oob_score_ : float
        The last value of the loss on the out-of-bag samples. It is
        the same as `oob_scores_[-1]`. Only available if `subsample < 1.0`.

        .. versionadded:: 1.3

    train_score_ : ndarray of shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the loss of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the loss on the training data.

    init_ : estimator
        The estimator that provides the initial predictions. Set via the ``init``
        argument.

    estimators_ : ndarray of DecisionTreeRegressor of shape (n_estimators, 1)
        The collection of fitted sub-estimators.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    max_features_ : int
        The inferred value of max_features.

    See Also
    --------
    HistGradientBoostingRegressor : Histogram-based Gradient Boosting
        Classification Tree.
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.
    sklearn.ensemble.RandomForestRegressor : A random forest regressor.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> reg = GradientBoostingRegressor(random_state=0)
    >>> reg.fit(X_train, y_train)
    GradientBoostingRegressor(random_state=0)
    >>> reg.predict(X_test[1:2])
    array([-61...])
    >>> reg.score(X_test, y_test)
    0.4...
    """
    _parameter_constraints: dict = {**BaseGradientBoosting._parameter_constraints, 'loss': [StrOptions({'squared_error', 'absolute_error', 'huber', 'quantile'})], 'init': [StrOptions({'zero'}), None, HasMethods(['fit', 'predict'])], 'alpha': [Interval(Real, 0.0, 1.0, closed='neither')]}

    def __init__(self, *, loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, init=init, subsample=subsample, max_features=max_features, min_impurity_decrease=min_impurity_decrease, random_state=random_state, alpha=alpha, verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

    def _encode_y(self, y=None, sample_weight=None):
        if False:
            while True:
                i = 10
        self.n_trees_per_iteration_ = 1
        y = y.astype(DOUBLE, copy=False)
        return y

    def _get_loss(self, sample_weight):
        if False:
            i = 10
            return i + 15
        if self.loss in ('quantile', 'huber'):
            return _LOSSES[self.loss](sample_weight=sample_weight, quantile=self.alpha)
        else:
            return _LOSSES[self.loss](sample_weight=sample_weight)

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Predict regression target for X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        Returns\n        -------\n        y : ndarray of shape (n_samples,)\n            The predicted values.\n        '
        X = self._validate_data(X, dtype=DTYPE, order='C', accept_sparse='csr', reset=False)
        return self._raw_predict(X).ravel()

    def staged_predict(self, X):
        if False:
            return 10
        'Predict regression target at each stage for X.\n\n        This method allows monitoring (i.e. determine error on testing set)\n        after each stage.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, it will be converted to\n            ``dtype=np.float32`` and if a sparse matrix is provided\n            to a sparse ``csr_matrix``.\n\n        Yields\n        ------\n        y : generator of ndarray of shape (n_samples,)\n            The predicted value of the input samples.\n        '
        for raw_predictions in self._staged_raw_predict(X):
            yield raw_predictions.ravel()

    def apply(self, X):
        if False:
            i = 10
            return i + 15
        'Apply trees in the ensemble to X, return leaf indices.\n\n        .. versionadded:: 0.17\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input samples. Internally, its dtype will be converted to\n            ``dtype=np.float32``. If a sparse matrix is provided, it will\n            be converted to a sparse ``csr_matrix``.\n\n        Returns\n        -------\n        X_leaves : array-like of shape (n_samples, n_estimators)\n            For each datapoint x in X and for each tree in the ensemble,\n            return the index of the leaf x ends up in each estimator.\n        '
        leaves = super().apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves