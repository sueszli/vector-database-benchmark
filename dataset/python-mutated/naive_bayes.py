"""
The :mod:`sklearn.naive_bayes` module implements Naive Bayes algorithms. These
are supervised learning methods based on applying Bayes' theorem with strong
(naive) feature independence assumptions.
"""
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import logsumexp
from .base import BaseEstimator, ClassifierMixin, _fit_context
from .preprocessing import LabelBinarizer, binarize, label_binarize
from .utils._param_validation import Hidden, Interval, StrOptions
from .utils.extmath import safe_sparse_dot
from .utils.multiclass import _check_partial_fit_first_call
from .utils.validation import _check_sample_weight, check_is_fitted, check_non_negative
__all__ = ['BernoulliNB', 'GaussianNB', 'MultinomialNB', 'ComplementNB', 'CategoricalNB']

class _BaseNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for naive Bayes estimators"""

    @abstractmethod
    def _joint_log_likelihood(self, X):
        if False:
            i = 10
            return i + 15
        'Compute the unnormalized posterior log probability of X\n\n        I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of\n        shape (n_samples, n_classes).\n\n        Public methods predict, predict_proba, predict_log_proba, and\n        predict_joint_log_proba pass the input through _check_X before handing it\n        over to _joint_log_likelihood. The term "joint log likelihood" is used\n        interchangibly with "joint log probability".\n        '

    @abstractmethod
    def _check_X(self, X):
        if False:
            return 10
        'To be overridden in subclasses with the actual checks.\n\n        Only used in predict* methods.\n        '

    def predict_joint_log_proba(self, X):
        if False:
            while True:
                i = 10
        'Return joint log probability estimates for the test vector X.\n\n        For each row x of X and class y, the joint log probability is given by\n        ``log P(x, y) = log P(y) + log P(x|y),``\n        where ``log P(y)`` is the class prior probability and ``log P(x|y)`` is\n        the class-conditional probability.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The input samples.\n\n        Returns\n        -------\n        C : ndarray of shape (n_samples, n_classes)\n            Returns the joint log-probability of the samples for each class in\n            the model. The columns correspond to the classes in sorted\n            order, as they appear in the attribute :term:`classes_`.\n        '
        check_is_fitted(self)
        X = self._check_X(X)
        return self._joint_log_likelihood(X)

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform classification on an array of test vectors X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The input samples.\n\n        Returns\n        -------\n        C : ndarray of shape (n_samples,)\n            Predicted target values for X.\n        '
        check_is_fitted(self)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        if False:
            return 10
        '\n        Return log-probability estimates for the test vector X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The input samples.\n\n        Returns\n        -------\n        C : array-like of shape (n_samples, n_classes)\n            Returns the log-probability of the samples for each class in\n            the model. The columns correspond to the classes in sorted\n            order, as they appear in the attribute :term:`classes_`.\n        '
        check_is_fitted(self)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        if False:
            return 10
        '\n        Return probability estimates for the test vector X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The input samples.\n\n        Returns\n        -------\n        C : array-like of shape (n_samples, n_classes)\n            Returns the probability of the samples for each class in\n            the model. The columns correspond to the classes in sorted\n            order, as they appear in the attribute :term:`classes_`.\n        '
        return np.exp(self.predict_log_proba(X))

class GaussianNB(_BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB).

    Can perform online updates to model parameters via :meth:`partial_fit`.
    For details on algorithm used to update feature means and variance online,
    see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

    Read more in the :ref:`User Guide <gaussian_naive_bayes>`.

    Parameters
    ----------
    priors : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.

        .. versionadded:: 0.20

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        number of training samples observed in each class.

    class_prior_ : ndarray of shape (n_classes,)
        probability of each class.

    classes_ : ndarray of shape (n_classes,)
        class labels known to the classifier.

    epsilon_ : float
        absolute additive value to variances.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    var_ : ndarray of shape (n_classes, n_features)
        Variance of each feature per class.

        .. versionadded:: 1.0

    theta_ : ndarray of shape (n_classes, n_features)
        mean of each feature per class.

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    CategoricalNB : Naive Bayes classifier for categorical features.
    ComplementNB : Complement Naive Bayes classifier.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    >>> clf_pf = GaussianNB()
    >>> clf_pf.partial_fit(X, Y, np.unique(Y))
    GaussianNB()
    >>> print(clf_pf.predict([[-0.8, -1]]))
    [1]
    """
    _parameter_constraints: dict = {'priors': ['array-like', None], 'var_smoothing': [Interval(Real, 0, None, closed='left')]}

    def __init__(self, *, priors=None, var_smoothing=1e-09):
        if False:
            while True:
                i = 10
        self.priors = priors
        self.var_smoothing = var_smoothing

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        if False:
            return 10
        'Fit Gaussian Naive Bayes according to X, y.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training vectors, where `n_samples` is the number of samples\n            and `n_features` is the number of features.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Weights applied to individual samples (1. for unweighted).\n\n            .. versionadded:: 0.17\n               Gaussian Naive Bayes supports fitting with *sample_weight*.\n\n        Returns\n        -------\n        self : object\n            Returns the instance itself.\n        '
        y = self._validate_data(y=y)
        return self._partial_fit(X, y, np.unique(y), _refit=True, sample_weight=sample_weight)

    def _check_X(self, X):
        if False:
            print('Hello World!')
        'Validate X, used only in predict* methods.'
        return self._validate_data(X, reset=False)

    @staticmethod
    def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
        if False:
            while True:
                i = 10
        'Compute online update of Gaussian mean and variance.\n\n        Given starting sample count, mean, and variance, a new set of\n        points X, and optionally sample weights, return the updated mean and\n        variance. (NB - each dimension (column) in X is treated as independent\n        -- you get variance, not covariance).\n\n        Can take scalar mean and variance, or vector mean and variance to\n        simultaneously update a number of independent Gaussians.\n\n        See Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:\n\n        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf\n\n        Parameters\n        ----------\n        n_past : int\n            Number of samples represented in old mean and variance. If sample\n            weights were given, this should contain the sum of sample\n            weights represented in old mean and variance.\n\n        mu : array-like of shape (number of Gaussians,)\n            Means for Gaussians in original set.\n\n        var : array-like of shape (number of Gaussians,)\n            Variances for Gaussians in original set.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Weights applied to individual samples (1. for unweighted).\n\n        Returns\n        -------\n        total_mu : array-like of shape (number of Gaussians,)\n            Updated mean for each Gaussian over the combined set.\n\n        total_var : array-like of shape (number of Gaussians,)\n            Updated variance for each Gaussian over the combined set.\n        '
        if X.shape[0] == 0:
            return (mu, var)
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            if np.isclose(n_new, 0.0):
                return (mu, var)
            new_mu = np.average(X, axis=0, weights=sample_weight)
            new_var = np.average((X - new_mu) ** 2, axis=0, weights=sample_weight)
        else:
            n_new = X.shape[0]
            new_var = np.var(X, axis=0)
            new_mu = np.mean(X, axis=0)
        if n_past == 0:
            return (new_mu, new_var)
        n_total = float(n_past + n_new)
        total_mu = (n_new * new_mu + n_past * mu) / n_total
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = old_ssd + new_ssd + n_new * n_past / n_total * (mu - new_mu) ** 2
        total_var = total_ssd / n_total
        return (total_mu, total_var)

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if False:
            i = 10
            return i + 15
        'Incremental fit on a batch of samples.\n\n        This method is expected to be called several times consecutively\n        on different chunks of a dataset so as to implement out-of-core\n        or online learning.\n\n        This is especially useful when the whole dataset is too big to fit in\n        memory at once.\n\n        This method has some performance and numerical stability overhead,\n        hence it is better to call partial_fit on chunks of data that are\n        as large as possible (as long as fitting in the memory budget) to\n        hide the overhead.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training vectors, where `n_samples` is the number of samples and\n            `n_features` is the number of features.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        classes : array-like of shape (n_classes,), default=None\n            List of all the classes that can possibly appear in the y vector.\n\n            Must be provided at the first call to partial_fit, can be omitted\n            in subsequent calls.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Weights applied to individual samples (1. for unweighted).\n\n            .. versionadded:: 0.17\n\n        Returns\n        -------\n        self : object\n            Returns the instance itself.\n        '
        return self._partial_fit(X, y, classes, _refit=False, sample_weight=sample_weight)

    def _partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):
        if False:
            while True:
                i = 10
        'Actual implementation of Gaussian NB fitting.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training vectors, where `n_samples` is the number of samples and\n            `n_features` is the number of features.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        classes : array-like of shape (n_classes,), default=None\n            List of all the classes that can possibly appear in the y vector.\n\n            Must be provided at the first call to partial_fit, can be omitted\n            in subsequent calls.\n\n        _refit : bool, default=False\n            If true, act as though this were the first time we called\n            _partial_fit (ie, throw away any past fitting and start over).\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Weights applied to individual samples (1. for unweighted).\n\n        Returns\n        -------\n        self : object\n        '
        if _refit:
            self.classes_ = None
        first_call = _check_partial_fit_first_call(self, classes)
        (X, y) = self._validate_data(X, y, reset=first_call)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        if first_call:
            n_features = X.shape[1]
            n_classes = len(self.classes_)
            self.theta_ = np.zeros((n_classes, n_features))
            self.var_ = np.zeros((n_classes, n_features))
            self.class_count_ = np.zeros(n_classes, dtype=np.float64)
            if self.priors is not None:
                priors = np.asarray(self.priors)
                if len(priors) != n_classes:
                    raise ValueError('Number of priors must match number of classes.')
                if not np.isclose(priors.sum(), 1.0):
                    raise ValueError('The sum of the priors should be 1.')
                if (priors < 0).any():
                    raise ValueError('Priors must be non-negative.')
                self.class_prior_ = priors
            else:
                self.class_prior_ = np.zeros(len(self.classes_), dtype=np.float64)
        else:
            if X.shape[1] != self.theta_.shape[1]:
                msg = 'Number of features %d does not match previous data %d.'
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
            self.var_[:, :] -= self.epsilon_
        classes = self.classes_
        unique_y = np.unique(y)
        unique_y_in_classes = np.isin(unique_y, classes)
        if not np.all(unique_y_in_classes):
            raise ValueError('The target label(s) %s in y do not exist in the initial classes %s' % (unique_y[~unique_y_in_classes], classes))
        for y_i in unique_y:
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]
            if sample_weight is not None:
                sw_i = sample_weight[y == y_i]
                N_i = sw_i.sum()
            else:
                sw_i = None
                N_i = X_i.shape[0]
            (new_theta, new_sigma) = self._update_mean_variance(self.class_count_[i], self.theta_[i, :], self.var_[i, :], X_i, sw_i)
            self.theta_[i, :] = new_theta
            self.var_[i, :] = new_sigma
            self.class_count_[i] += N_i
        self.var_[:, :] += self.epsilon_
        if self.priors is None:
            self.class_prior_ = self.class_count_ / self.class_count_.sum()
        return self

    def _joint_log_likelihood(self, X):
        if False:
            print('Hello World!')
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i, :]))
            n_ij -= 0.5 * np.sum((X - self.theta_[i, :]) ** 2 / self.var_[i, :], 1)
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

class _BaseDiscreteNB(_BaseNB):
    """Abstract base class for naive Bayes on discrete/categorical data

    Any estimator based on this class should provide:

    __init__
    _joint_log_likelihood(X) as per _BaseNB
    _update_feature_log_prob(alpha)
    _count(X, Y)
    """
    _parameter_constraints: dict = {'alpha': [Interval(Real, 0, None, closed='left'), 'array-like'], 'fit_prior': ['boolean'], 'class_prior': ['array-like', None], 'force_alpha': ['boolean', Hidden(StrOptions({'warn'}))]}

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, force_alpha='warn'):
        if False:
            i = 10
            return i + 15
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.force_alpha = force_alpha

    @abstractmethod
    def _count(self, X, Y):
        if False:
            print('Hello World!')
        'Update counts that are used to calculate probabilities.\n\n        The counts make up a sufficient statistic extracted from the data.\n        Accordingly, this method is called each time `fit` or `partial_fit`\n        update the model. `class_count_` and `feature_count_` must be updated\n        here along with any model specific counts.\n\n        Parameters\n        ----------\n        X : {ndarray, sparse matrix} of shape (n_samples, n_features)\n            The input samples.\n        Y : ndarray of shape (n_samples, n_classes)\n            Binarized class labels.\n        '

    @abstractmethod
    def _update_feature_log_prob(self, alpha):
        if False:
            for i in range(10):
                print('nop')
        'Update feature log probabilities based on counts.\n\n        This method is called each time `fit` or `partial_fit` update the\n        model.\n\n        Parameters\n        ----------\n        alpha : float\n            smoothing parameter. See :meth:`_check_alpha`.\n        '

    def _check_X(self, X):
        if False:
            i = 10
            return i + 15
        'Validate X, used only in predict* methods.'
        return self._validate_data(X, accept_sparse='csr', reset=False)

    def _check_X_y(self, X, y, reset=True):
        if False:
            while True:
                i = 10
        'Validate X and y in fit methods.'
        return self._validate_data(X, y, accept_sparse='csr', reset=reset)

    def _update_class_log_prior(self, class_prior=None):
        if False:
            i = 10
            return i + 15
        'Update class log priors.\n\n        The class log priors are based on `class_prior`, class count or the\n        number of classes. This method is called each time `fit` or\n        `partial_fit` update the model.\n        '
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError('Number of priors must match number of classes.')
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                log_class_count = np.log(self.class_count_)
            self.class_log_prior_ = log_class_count - np.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

    def _check_alpha(self):
        if False:
            i = 10
            return i + 15
        alpha = np.asarray(self.alpha) if not isinstance(self.alpha, Real) else self.alpha
        alpha_min = np.min(alpha)
        if isinstance(alpha, np.ndarray):
            if not alpha.shape[0] == self.n_features_in_:
                raise ValueError(f'When alpha is an array, it should contains `n_features`. Got {alpha.shape[0]} elements instead of {self.n_features_in_}.')
            if alpha_min < 0:
                raise ValueError('All values in alpha must be greater than 0.')
        alpha_lower_bound = 1e-10
        _force_alpha = self.force_alpha
        if _force_alpha == 'warn' and alpha_min < alpha_lower_bound:
            _force_alpha = False
            warnings.warn('The default value for `force_alpha` will change to `True` in 1.4. To suppress this warning, manually set the value of `force_alpha`.', FutureWarning)
        if alpha_min < alpha_lower_bound and (not _force_alpha):
            warnings.warn(f'alpha too small will result in numeric errors, setting alpha = {alpha_lower_bound:.1e}. Use `force_alpha=True` to keep alpha unchanged.')
            return np.maximum(alpha, alpha_lower_bound)
        return alpha

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if False:
            i = 10
            return i + 15
        'Incremental fit on a batch of samples.\n\n        This method is expected to be called several times consecutively\n        on different chunks of a dataset so as to implement out-of-core\n        or online learning.\n\n        This is especially useful when the whole dataset is too big to fit in\n        memory at once.\n\n        This method has some performance overhead hence it is better to call\n        partial_fit on chunks of data that are as large as possible\n        (as long as fitting in the memory budget) to hide the overhead.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training vectors, where `n_samples` is the number of samples and\n            `n_features` is the number of features.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        classes : array-like of shape (n_classes,), default=None\n            List of all the classes that can possibly appear in the y vector.\n\n            Must be provided at the first call to partial_fit, can be omitted\n            in subsequent calls.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Weights applied to individual samples (1. for unweighted).\n\n        Returns\n        -------\n        self : object\n            Returns the instance itself.\n        '
        first_call = not hasattr(self, 'classes_')
        (X, y) = self._check_X_y(X, y, reset=first_call)
        (_, n_features) = X.shape
        if _check_partial_fit_first_call(self, classes):
            n_classes = len(classes)
            self._init_counters(n_classes, n_features)
        Y = label_binarize(y, classes=self.classes_)
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:
                Y = np.ones_like(Y)
        if X.shape[0] != Y.shape[0]:
            msg = 'X.shape[0]=%d and y.shape[0]=%d are incompatible.'
            raise ValueError(msg % (X.shape[0], y.shape[0]))
        Y = Y.astype(np.float64, copy=False)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T
        class_prior = self.class_prior
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        if False:
            while True:
                i = 10
        'Fit Naive Bayes classifier according to X, y.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training vectors, where `n_samples` is the number of samples and\n            `n_features` is the number of features.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Weights applied to individual samples (1. for unweighted).\n\n        Returns\n        -------\n        self : object\n            Returns the instance itself.\n        '
        (X, y) = self._check_X_y(X, y)
        (_, n_features) = X.shape
        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:
                Y = np.ones_like(Y)
        if sample_weight is not None:
            Y = Y.astype(np.float64, copy=False)
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T
        class_prior = self.class_prior
        n_classes = Y.shape[1]
        self._init_counters(n_classes, n_features)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    def _init_counters(self, n_classes, n_features):
        if False:
            i = 10
            return i + 15
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)

    def _more_tags(self):
        if False:
            return 10
        return {'poor_score': True}

class MultinomialNB(_BaseDiscreteNB):
    """
    Naive Bayes classifier for multinomial models.

    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.

    Read more in the :ref:`User Guide <multinomial_naive_bayes>`.

    Parameters
    ----------
    alpha : float or array-like of shape (n_features,), default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).

    force_alpha : bool, default=False
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.

        .. versionadded:: 1.2
        .. deprecated:: 1.2
           The default value of `force_alpha` will change to `True` in v1.4.

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features
        given a class, ``P(x_i|y)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    CategoricalNB : Naive Bayes classifier for categorical features.
    ComplementNB : Complement Naive Bayes classifier.
    GaussianNB : Gaussian Naive Bayes.

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB(force_alpha=True)
    >>> clf.fit(X, y)
    MultinomialNB(force_alpha=True)
    >>> print(clf.predict(X[2:3]))
    [3]
    """

    def __init__(self, *, alpha=1.0, force_alpha='warn', fit_prior=True, class_prior=None):
        if False:
            while True:
                i = 10
        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior, force_alpha=force_alpha)

    def _more_tags(self):
        if False:
            print('Hello World!')
        return {'requires_positive_X': True}

    def _count(self, X, Y):
        if False:
            while True:
                i = 10
        'Count and smooth feature occurrences.'
        check_non_negative(X, 'MultinomialNB (input X)')
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        if False:
            print('Hello World!')
        'Apply smoothing to raw counts and recompute log probabilities'
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))

    def _joint_log_likelihood(self, X):
        if False:
            i = 10
            return i + 15
        'Calculate the posterior log probability of the samples X'
        return safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_

class ComplementNB(_BaseDiscreteNB):
    """The Complement Naive Bayes classifier described in Rennie et al. (2003).

    The Complement Naive Bayes classifier was designed to correct the "severe
    assumptions" made by the standard Multinomial Naive Bayes classifier. It is
    particularly suited for imbalanced data sets.

    Read more in the :ref:`User Guide <complement_naive_bayes>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    alpha : float or array-like of shape (n_features,), default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).

    force_alpha : bool, default=False
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.

        .. versionadded:: 1.2
        .. deprecated:: 1.2
           The default value of `force_alpha` will change to `True` in v1.4.

    fit_prior : bool, default=True
        Only used in edge case with a single class in the training set.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. Not used.

    norm : bool, default=False
        Whether or not a second normalization of the weights is performed. The
        default behavior mirrors the implementations found in Mahout and Weka,
        which do not follow the full algorithm described in Table 9 of the
        paper.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class. Only used in edge
        case with a single class in the training set.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_all_ : ndarray of shape (n_features,)
        Number of samples encountered for each feature during fitting. This
        value is weighted by the sample weight when provided.

    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature) during fitting.
        This value is weighted by the sample weight when provided.

    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical weights for class complements.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    CategoricalNB : Naive Bayes classifier for categorical features.
    GaussianNB : Gaussian Naive Bayes.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    References
    ----------
    Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003).
    Tackling the poor assumptions of naive bayes text classifiers. In ICML
    (Vol. 3, pp. 616-623).
    https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import ComplementNB
    >>> clf = ComplementNB(force_alpha=True)
    >>> clf.fit(X, y)
    ComplementNB(force_alpha=True)
    >>> print(clf.predict(X[2:3]))
    [3]
    """
    _parameter_constraints: dict = {**_BaseDiscreteNB._parameter_constraints, 'norm': ['boolean']}

    def __init__(self, *, alpha=1.0, force_alpha='warn', fit_prior=True, class_prior=None, norm=False):
        if False:
            while True:
                i = 10
        super().__init__(alpha=alpha, force_alpha=force_alpha, fit_prior=fit_prior, class_prior=class_prior)
        self.norm = norm

    def _more_tags(self):
        if False:
            while True:
                i = 10
        return {'requires_positive_X': True}

    def _count(self, X, Y):
        if False:
            while True:
                i = 10
        'Count feature occurrences.'
        check_non_negative(X, 'ComplementNB (input X)')
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)
        self.feature_all_ = self.feature_count_.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        if False:
            print('Hello World!')
        'Apply smoothing to raw counts and compute the weights.'
        comp_count = self.feature_all_ + alpha - self.feature_count_
        logged = np.log(comp_count / comp_count.sum(axis=1, keepdims=True))
        if self.norm:
            summed = logged.sum(axis=1, keepdims=True)
            feature_log_prob = logged / summed
        else:
            feature_log_prob = -logged
        self.feature_log_prob_ = feature_log_prob

    def _joint_log_likelihood(self, X):
        if False:
            return 10
        'Calculate the class scores for the samples in X.'
        jll = safe_sparse_dot(X, self.feature_log_prob_.T)
        if len(self.classes_) == 1:
            jll += self.class_log_prior_
        return jll

class BernoulliNB(_BaseDiscreteNB):
    """Naive Bayes classifier for multivariate Bernoulli models.

    Like MultinomialNB, this classifier is suitable for discrete data. The
    difference is that while MultinomialNB works with occurrence counts,
    BernoulliNB is designed for binary/boolean features.

    Read more in the :ref:`User Guide <bernoulli_naive_bayes>`.

    Parameters
    ----------
    alpha : float or array-like of shape (n_features,), default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).

    force_alpha : bool, default=False
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.

        .. versionadded:: 1.2
        .. deprecated:: 1.2
           The default value of `force_alpha` will change to `True` in v1.4.

    binarize : float or None, default=0.0
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Log probability of each class (smoothed).

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    CategoricalNB : Naive Bayes classifier for categorical features.
    ComplementNB : The Complement Naive Bayes classifier
        described in Rennie et al. (2003).
    GaussianNB : Gaussian Naive Bayes (GaussianNB).
    MultinomialNB : Naive Bayes classifier for multinomial models.

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

    A. McCallum and K. Nigam (1998). A comparison of event models for naive
    Bayes text classification. Proc. AAAI/ICML-98 Workshop on Learning for
    Text Categorization, pp. 41-48.

    V. Metsis, I. Androutsopoulos and G. Paliouras (2006). Spam filtering with
    naive Bayes -- Which naive Bayes? 3rd Conf. on Email and Anti-Spam (CEAS).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 4, 5])
    >>> from sklearn.naive_bayes import BernoulliNB
    >>> clf = BernoulliNB(force_alpha=True)
    >>> clf.fit(X, Y)
    BernoulliNB(force_alpha=True)
    >>> print(clf.predict(X[2:3]))
    [3]
    """
    _parameter_constraints: dict = {**_BaseDiscreteNB._parameter_constraints, 'binarize': [None, Interval(Real, 0, None, closed='left')]}

    def __init__(self, *, alpha=1.0, force_alpha='warn', binarize=0.0, fit_prior=True, class_prior=None):
        if False:
            return 10
        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior, force_alpha=force_alpha)
        self.binarize = binarize

    def _check_X(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Validate X, used only in predict* methods.'
        X = super()._check_X(X)
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        return X

    def _check_X_y(self, X, y, reset=True):
        if False:
            while True:
                i = 10
        (X, y) = super()._check_X_y(X, y, reset=reset)
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        return (X, y)

    def _count(self, X, Y):
        if False:
            while True:
                i = 10
        'Count and smooth feature occurrences.'
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        if False:
            print('Hello World!')
        'Apply smoothing to raw counts and recompute log probabilities'
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = self.class_count_ + alpha * 2
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))

    def _joint_log_likelihood(self, X):
        if False:
            return 10
        'Calculate the posterior log probability of the samples X'
        n_features = self.feature_log_prob_.shape[1]
        n_features_X = X.shape[1]
        if n_features_X != n_features:
            raise ValueError('Expected input with %d features, got %d instead' % (n_features, n_features_X))
        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)
        return jll

class CategoricalNB(_BaseDiscreteNB):
    """Naive Bayes classifier for categorical features.

    The categorical Naive Bayes classifier is suitable for classification with
    discrete features that are categorically distributed. The categories of
    each feature are drawn from a categorical distribution.

    Read more in the :ref:`User Guide <categorical_naive_bayes>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).

    force_alpha : bool, default=False
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.

        .. versionadded:: 1.2
        .. deprecated:: 1.2
           The default value of `force_alpha` will change to `True` in v1.4.

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    min_categories : int or array-like of shape (n_features,), default=None
        Minimum number of categories per feature.

        - integer: Sets the minimum number of categories per feature to
          `n_categories` for each features.
        - array-like: shape (n_features,) where `n_categories[i]` holds the
          minimum number of categories for the ith column of the input.
        - None (default): Determines the number of categories automatically
          from the training data.

        .. versionadded:: 0.24

    Attributes
    ----------
    category_count_ : list of arrays of shape (n_features,)
        Holds arrays of shape (n_classes, n_categories of respective feature)
        for each feature. Each array provides the number of samples
        encountered for each class and category of the specific feature.

    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_log_prob_ : list of arrays of shape (n_features,)
        Holds arrays of shape (n_classes, n_categories of respective feature)
        for each feature. Each array provides the empirical log probability
        of categories given the respective feature and class, ``P(x_i|y)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_categories_ : ndarray of shape (n_features,), dtype=np.int64
        Number of categories for each feature. This value is
        inferred from the data or set by the minimum number of categories.

        .. versionadded:: 0.24

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    ComplementNB : Complement Naive Bayes classifier.
    GaussianNB : Gaussian Naive Bayes.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import CategoricalNB
    >>> clf = CategoricalNB(force_alpha=True)
    >>> clf.fit(X, y)
    CategoricalNB(force_alpha=True)
    >>> print(clf.predict(X[2:3]))
    [3]
    """
    _parameter_constraints: dict = {**_BaseDiscreteNB._parameter_constraints, 'min_categories': [None, 'array-like', Interval(Integral, 1, None, closed='left')], 'alpha': [Interval(Real, 0, None, closed='left')]}

    def __init__(self, *, alpha=1.0, force_alpha='warn', fit_prior=True, class_prior=None, min_categories=None):
        if False:
            i = 10
            return i + 15
        super().__init__(alpha=alpha, force_alpha=force_alpha, fit_prior=fit_prior, class_prior=class_prior)
        self.min_categories = min_categories

    def fit(self, X, y, sample_weight=None):
        if False:
            print('Hello World!')
        'Fit Naive Bayes classifier according to X, y.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training vectors, where `n_samples` is the number of samples and\n            `n_features` is the number of features. Here, each feature of X is\n            assumed to be from a different categorical distribution.\n            It is further assumed that all categories of each feature are\n            represented by the numbers 0, ..., n - 1, where n refers to the\n            total number of categories for the given feature. This can, for\n            instance, be achieved with the help of OrdinalEncoder.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Weights applied to individual samples (1. for unweighted).\n\n        Returns\n        -------\n        self : object\n            Returns the instance itself.\n        '
        return super().fit(X, y, sample_weight=sample_weight)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if False:
            while True:
                i = 10
        'Incremental fit on a batch of samples.\n\n        This method is expected to be called several times consecutively\n        on different chunks of a dataset so as to implement out-of-core\n        or online learning.\n\n        This is especially useful when the whole dataset is too big to fit in\n        memory at once.\n\n        This method has some performance overhead hence it is better to call\n        partial_fit on chunks of data that are as large as possible\n        (as long as fitting in the memory budget) to hide the overhead.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training vectors, where `n_samples` is the number of samples and\n            `n_features` is the number of features. Here, each feature of X is\n            assumed to be from a different categorical distribution.\n            It is further assumed that all categories of each feature are\n            represented by the numbers 0, ..., n - 1, where n refers to the\n            total number of categories for the given feature. This can, for\n            instance, be achieved with the help of OrdinalEncoder.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        classes : array-like of shape (n_classes,), default=None\n            List of all the classes that can possibly appear in the y vector.\n\n            Must be provided at the first call to partial_fit, can be omitted\n            in subsequent calls.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Weights applied to individual samples (1. for unweighted).\n\n        Returns\n        -------\n        self : object\n            Returns the instance itself.\n        '
        return super().partial_fit(X, y, classes, sample_weight=sample_weight)

    def _more_tags(self):
        if False:
            print('Hello World!')
        return {'requires_positive_X': True}

    def _check_X(self, X):
        if False:
            print('Hello World!')
        'Validate X, used only in predict* methods.'
        X = self._validate_data(X, dtype='int', accept_sparse=False, force_all_finite=True, reset=False)
        check_non_negative(X, 'CategoricalNB (input X)')
        return X

    def _check_X_y(self, X, y, reset=True):
        if False:
            while True:
                i = 10
        (X, y) = self._validate_data(X, y, dtype='int', accept_sparse=False, force_all_finite=True, reset=reset)
        check_non_negative(X, 'CategoricalNB (input X)')
        return (X, y)

    def _init_counters(self, n_classes, n_features):
        if False:
            for i in range(10):
                print('nop')
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.category_count_ = [np.zeros((n_classes, 0)) for _ in range(n_features)]

    @staticmethod
    def _validate_n_categories(X, min_categories):
        if False:
            for i in range(10):
                print('nop')
        n_categories_X = X.max(axis=0) + 1
        min_categories_ = np.array(min_categories)
        if min_categories is not None:
            if not np.issubdtype(min_categories_.dtype, np.signedinteger):
                raise ValueError(f"'min_categories' should have integral type. Got {min_categories_.dtype} instead.")
            n_categories_ = np.maximum(n_categories_X, min_categories_, dtype=np.int64)
            if n_categories_.shape != n_categories_X.shape:
                raise ValueError(f"'min_categories' should have shape ({X.shape[1]},) when an array-like is provided. Got {min_categories_.shape} instead.")
            return n_categories_
        else:
            return n_categories_X

    def _count(self, X, Y):
        if False:
            while True:
                i = 10

        def _update_cat_count_dims(cat_count, highest_feature):
            if False:
                i = 10
                return i + 15
            diff = highest_feature + 1 - cat_count.shape[1]
            if diff > 0:
                return np.pad(cat_count, [(0, 0), (0, diff)], 'constant')
            return cat_count

        def _update_cat_count(X_feature, Y, cat_count, n_classes):
            if False:
                for i in range(10):
                    print('nop')
            for j in range(n_classes):
                mask = Y[:, j].astype(bool)
                if Y.dtype.type == np.int64:
                    weights = None
                else:
                    weights = Y[mask, j]
                counts = np.bincount(X_feature[mask], weights=weights)
                indices = np.nonzero(counts)[0]
                cat_count[j, indices] += counts[indices]
        self.class_count_ += Y.sum(axis=0)
        self.n_categories_ = self._validate_n_categories(X, self.min_categories)
        for i in range(self.n_features_in_):
            X_feature = X[:, i]
            self.category_count_[i] = _update_cat_count_dims(self.category_count_[i], self.n_categories_[i] - 1)
            _update_cat_count(X_feature, Y, self.category_count_[i], self.class_count_.shape[0])

    def _update_feature_log_prob(self, alpha):
        if False:
            i = 10
            return i + 15
        feature_log_prob = []
        for i in range(self.n_features_in_):
            smoothed_cat_count = self.category_count_[i] + alpha
            smoothed_class_count = smoothed_cat_count.sum(axis=1)
            feature_log_prob.append(np.log(smoothed_cat_count) - np.log(smoothed_class_count.reshape(-1, 1)))
        self.feature_log_prob_ = feature_log_prob

    def _joint_log_likelihood(self, X):
        if False:
            return 10
        self._check_n_features(X, reset=False)
        jll = np.zeros((X.shape[0], self.class_count_.shape[0]))
        for i in range(self.n_features_in_):
            indices = X[:, i]
            jll += self.feature_log_prob_[i][:, indices].T
        total_ll = jll + self.class_log_prior_
        return total_ll