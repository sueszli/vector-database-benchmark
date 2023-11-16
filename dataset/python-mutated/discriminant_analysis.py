"""
Linear Discriminant Analysis and Quadratic Discriminant Analysis
"""
import warnings
from numbers import Integral, Real
import numpy as np
import scipy.linalg
from scipy import linalg
from .base import BaseEstimator, ClassifierMixin, ClassNamePrefixFeaturesOutMixin, TransformerMixin, _fit_context
from .covariance import empirical_covariance, ledoit_wolf, shrunk_covariance
from .linear_model._base import LinearClassifierMixin
from .preprocessing import StandardScaler
from .utils._array_api import _expit, device, get_namespace, size
from .utils._param_validation import HasMethods, Interval, StrOptions
from .utils.extmath import softmax
from .utils.multiclass import check_classification_targets, unique_labels
from .utils.validation import check_is_fitted
__all__ = ['LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis']

def _cov(X, shrinkage=None, covariance_estimator=None):
    if False:
        print('Hello World!')
    "Estimate covariance matrix (using optional covariance_estimator).\n    Parameters\n    ----------\n    X : array-like of shape (n_samples, n_features)\n        Input data.\n\n    shrinkage : {'empirical', 'auto'} or float, default=None\n        Shrinkage parameter, possible values:\n          - None or 'empirical': no shrinkage (default).\n          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.\n          - float between 0 and 1: fixed shrinkage parameter.\n\n        Shrinkage parameter is ignored if  `covariance_estimator`\n        is not None.\n\n    covariance_estimator : estimator, default=None\n        If not None, `covariance_estimator` is used to estimate\n        the covariance matrices instead of relying on the empirical\n        covariance estimator (with potential shrinkage).\n        The object should have a fit method and a ``covariance_`` attribute\n        like the estimators in :mod:`sklearn.covariance``.\n        if None the shrinkage parameter drives the estimate.\n\n        .. versionadded:: 0.24\n\n    Returns\n    -------\n    s : ndarray of shape (n_features, n_features)\n        Estimated covariance matrix.\n    "
    if covariance_estimator is None:
        shrinkage = 'empirical' if shrinkage is None else shrinkage
        if isinstance(shrinkage, str):
            if shrinkage == 'auto':
                sc = StandardScaler()
                X = sc.fit_transform(X)
                s = ledoit_wolf(X)[0]
                s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            elif shrinkage == 'empirical':
                s = empirical_covariance(X)
        elif isinstance(shrinkage, Real):
            s = shrunk_covariance(empirical_covariance(X), shrinkage)
    else:
        if shrinkage is not None and shrinkage != 0:
            raise ValueError('covariance_estimator and shrinkage parameters are not None. Only one of the two can be set.')
        covariance_estimator.fit(X)
        if not hasattr(covariance_estimator, 'covariance_'):
            raise ValueError('%s does not have a covariance_ attribute' % covariance_estimator.__class__.__name__)
        s = covariance_estimator.covariance_
    return s

def _class_means(X, y):
    if False:
        for i in range(10):
            print('nop')
    'Compute class means.\n\n    Parameters\n    ----------\n    X : array-like of shape (n_samples, n_features)\n        Input data.\n\n    y : array-like of shape (n_samples,) or (n_samples, n_targets)\n        Target values.\n\n    Returns\n    -------\n    means : array-like of shape (n_classes, n_features)\n        Class means.\n    '
    (xp, is_array_api_compliant) = get_namespace(X)
    (classes, y) = xp.unique_inverse(y)
    means = xp.zeros((classes.shape[0], X.shape[1]), device=device(X), dtype=X.dtype)
    if is_array_api_compliant:
        for i in range(classes.shape[0]):
            means[i, :] = xp.mean(X[y == i], axis=0)
    else:
        cnt = np.bincount(y)
        np.add.at(means, y, X)
        means /= cnt[:, None]
    return means

def _class_cov(X, y, priors, shrinkage=None, covariance_estimator=None):
    if False:
        return 10
    "Compute weighted within-class covariance matrix.\n\n    The per-class covariance are weighted by the class priors.\n\n    Parameters\n    ----------\n    X : array-like of shape (n_samples, n_features)\n        Input data.\n\n    y : array-like of shape (n_samples,) or (n_samples, n_targets)\n        Target values.\n\n    priors : array-like of shape (n_classes,)\n        Class priors.\n\n    shrinkage : 'auto' or float, default=None\n        Shrinkage parameter, possible values:\n          - None: no shrinkage (default).\n          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.\n          - float between 0 and 1: fixed shrinkage parameter.\n\n        Shrinkage parameter is ignored if `covariance_estimator` is not None.\n\n    covariance_estimator : estimator, default=None\n        If not None, `covariance_estimator` is used to estimate\n        the covariance matrices instead of relying the empirical\n        covariance estimator (with potential shrinkage).\n        The object should have a fit method and a ``covariance_`` attribute\n        like the estimators in sklearn.covariance.\n        If None, the shrinkage parameter drives the estimate.\n\n        .. versionadded:: 0.24\n\n    Returns\n    -------\n    cov : array-like of shape (n_features, n_features)\n        Weighted within-class covariance matrix\n    "
    classes = np.unique(y)
    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for (idx, group) in enumerate(classes):
        Xg = X[y == group, :]
        cov += priors[idx] * np.atleast_2d(_cov(Xg, shrinkage, covariance_estimator))
    return cov

class LinearDiscriminantAnalysis(ClassNamePrefixFeaturesOutMixin, LinearClassifierMixin, TransformerMixin, BaseEstimator):
    """Linear Discriminant Analysis.

    A classifier with a linear decision boundary, generated by fitting class
    conditional densities to the data and using Bayes' rule.

    The model fits a Gaussian density to each class, assuming that all classes
    share the same covariance matrix.

    The fitted model can also be used to reduce the dimensionality of the input
    by projecting it to the most discriminative directions, using the
    `transform` method.

    .. versionadded:: 0.17
       *LinearDiscriminantAnalysis*.

    Read more in the :ref:`User Guide <lda_qda>`.

    Parameters
    ----------
    solver : {'svd', 'lsqr', 'eigen'}, default='svd'
        Solver to use, possible values:
          - 'svd': Singular value decomposition (default).
            Does not compute the covariance matrix, therefore this solver is
            recommended for data with a large number of features.
          - 'lsqr': Least squares solution.
            Can be combined with shrinkage or custom covariance estimator.
          - 'eigen': Eigenvalue decomposition.
            Can be combined with shrinkage or custom covariance estimator.

        .. versionchanged:: 1.2
            `solver="svd"` now has experimental Array API support. See the
            :ref:`Array API User Guide <array_api>` for more details.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        This should be left to None if `covariance_estimator` is used.
        Note that shrinkage works only with 'lsqr' and 'eigen' solvers.

    priors : array-like of shape (n_classes,), default=None
        The class prior probabilities. By default, the class proportions are
        inferred from the training data.

    n_components : int, default=None
        Number of components (<= min(n_classes - 1, n_features)) for
        dimensionality reduction. If None, will be set to
        min(n_classes - 1, n_features). This parameter only affects the
        `transform` method.

    store_covariance : bool, default=False
        If True, explicitly compute the weighted within-class covariance
        matrix when solver is 'svd'. The matrix is always computed
        and stored for the other solvers.

        .. versionadded:: 0.17

    tol : float, default=1.0e-4
        Absolute threshold for a singular value of X to be considered
        significant, used to estimate the rank of X. Dimensions whose
        singular values are non-significant are discarded. Only used if
        solver is 'svd'.

        .. versionadded:: 0.17

    covariance_estimator : covariance estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying on the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in :mod:`sklearn.covariance`.
        if None the shrinkage parameter drives the estimate.

        This should be left to None if `shrinkage` is used.
        Note that `covariance_estimator` works only with 'lsqr' and 'eigen'
        solvers.

        .. versionadded:: 0.24

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_classes, n_features)
        Weight vector(s).

    intercept_ : ndarray of shape (n_classes,)
        Intercept term.

    covariance_ : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix. It corresponds to
        `sum_k prior_k * C_k` where `C_k` is the covariance matrix of the
        samples in class `k`. The `C_k` are estimated using the (potentially
        shrunk) biased estimator of covariance. If solver is 'svd', only
        exists when `store_covariance` is True.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0. Only available when eigen
        or svd solver is used.

    means_ : array-like of shape (n_classes, n_features)
        Class-wise means.

    priors_ : array-like of shape (n_classes,)
        Class priors (sum to 1).

    scalings_ : array-like of shape (rank, n_classes - 1)
        Scaling of the features in the space spanned by the class centroids.
        Only available for 'svd' and 'eigen' solvers.

    xbar_ : array-like of shape (n_features,)
        Overall mean. Only present if solver is 'svd'.

    classes_ : array-like of shape (n_classes,)
        Unique class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    QuadraticDiscriminantAnalysis : Quadratic Discriminant Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = LinearDiscriminantAnalysis()
    >>> clf.fit(X, y)
    LinearDiscriminantAnalysis()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """
    _parameter_constraints: dict = {'solver': [StrOptions({'svd', 'lsqr', 'eigen'})], 'shrinkage': [StrOptions({'auto'}), Interval(Real, 0, 1, closed='both'), None], 'n_components': [Interval(Integral, 1, None, closed='left'), None], 'priors': ['array-like', None], 'store_covariance': ['boolean'], 'tol': [Interval(Real, 0, None, closed='left')], 'covariance_estimator': [HasMethods('fit'), None]}

    def __init__(self, solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001, covariance_estimator=None):
        if False:
            return 10
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol
        self.covariance_estimator = covariance_estimator

    def _solve_lstsq(self, X, y, shrinkage, covariance_estimator):
        if False:
            return 10
        "Least squares solver.\n\n        The least squares solver computes a straightforward solution of the\n        optimal decision rule based directly on the discriminant functions. It\n        can only be used for classification (with any covariance estimator),\n        because\n        estimation of eigenvectors is not performed. Therefore, dimensionality\n        reduction with the transform is not supported.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training data.\n\n        y : array-like of shape (n_samples,) or (n_samples, n_classes)\n            Target values.\n\n        shrinkage : 'auto', float or None\n            Shrinkage parameter, possible values:\n              - None: no shrinkage.\n              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.\n              - float between 0 and 1: fixed shrinkage parameter.\n\n            Shrinkage parameter is ignored if  `covariance_estimator` i\n            not None\n\n        covariance_estimator : estimator, default=None\n            If not None, `covariance_estimator` is used to estimate\n            the covariance matrices instead of relying the empirical\n            covariance estimator (with potential shrinkage).\n            The object should have a fit method and a ``covariance_`` attribute\n            like the estimators in sklearn.covariance.\n            if None the shrinkage parameter drives the estimate.\n\n            .. versionadded:: 0.24\n\n        Notes\n        -----\n        This solver is based on [1]_, section 2.6.2, pp. 39-41.\n\n        References\n        ----------\n        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification\n           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN\n           0-471-05669-3.\n        "
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_, shrinkage, covariance_estimator)
        self.coef_ = linalg.lstsq(self.covariance_, self.means_.T)[0].T
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_)

    def _solve_eigen(self, X, y, shrinkage, covariance_estimator):
        if False:
            print('Hello World!')
        "Eigenvalue solver.\n\n        The eigenvalue solver computes the optimal solution of the Rayleigh\n        coefficient (basically the ratio of between class scatter to within\n        class scatter). This solver supports both classification and\n        dimensionality reduction (with any covariance estimator).\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training data.\n\n        y : array-like of shape (n_samples,) or (n_samples, n_targets)\n            Target values.\n\n        shrinkage : 'auto', float or None\n            Shrinkage parameter, possible values:\n              - None: no shrinkage.\n              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.\n              - float between 0 and 1: fixed shrinkage constant.\n\n            Shrinkage parameter is ignored if  `covariance_estimator` i\n            not None\n\n        covariance_estimator : estimator, default=None\n            If not None, `covariance_estimator` is used to estimate\n            the covariance matrices instead of relying the empirical\n            covariance estimator (with potential shrinkage).\n            The object should have a fit method and a ``covariance_`` attribute\n            like the estimators in sklearn.covariance.\n            if None the shrinkage parameter drives the estimate.\n\n            .. versionadded:: 0.24\n\n        Notes\n        -----\n        This solver is based on [1]_, section 3.8.3, pp. 121-124.\n\n        References\n        ----------\n        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification\n           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN\n           0-471-05669-3.\n        "
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_, shrinkage, covariance_estimator)
        Sw = self.covariance_
        St = _cov(X, shrinkage, covariance_estimator)
        Sb = St - Sw
        (evals, evecs) = linalg.eigh(Sb, Sw)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][:self._max_components]
        evecs = evecs[:, np.argsort(evals)[::-1]]
        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_)

    def _solve_svd(self, X, y):
        if False:
            i = 10
            return i + 15
        'SVD solver.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training data.\n\n        y : array-like of shape (n_samples,) or (n_samples, n_targets)\n            Target values.\n        '
        (xp, is_array_api_compliant) = get_namespace(X)
        if is_array_api_compliant:
            svd = xp.linalg.svd
        else:
            svd = scipy.linalg.svd
        (n_samples, n_features) = X.shape
        n_classes = self.classes_.shape[0]
        self.means_ = _class_means(X, y)
        if self.store_covariance:
            self.covariance_ = _class_cov(X, y, self.priors_)
        Xc = []
        for (idx, group) in enumerate(self.classes_):
            Xg = X[y == group]
            Xc.append(Xg - self.means_[idx, :])
        self.xbar_ = self.priors_ @ self.means_
        Xc = xp.concat(Xc, axis=0)
        std = xp.std(Xc, axis=0)
        std[std == 0] = 1.0
        fac = xp.asarray(1.0 / (n_samples - n_classes))
        X = xp.sqrt(fac) * (Xc / std)
        (U, S, Vt) = svd(X, full_matrices=False)
        rank = xp.sum(xp.astype(S > self.tol, xp.int32))
        scalings = (Vt[:rank, :] / std).T / S[:rank]
        fac = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)
        X = (xp.sqrt(n_samples * self.priors_ * fac) * (self.means_ - self.xbar_).T).T @ scalings
        (_, S, Vt) = svd(X, full_matrices=False)
        if self._max_components == 0:
            self.explained_variance_ratio_ = xp.empty((0,), dtype=S.dtype)
        else:
            self.explained_variance_ratio_ = (S ** 2 / xp.sum(S ** 2))[:self._max_components]
        rank = xp.sum(xp.astype(S > self.tol * S[0], xp.int32))
        self.scalings_ = scalings @ Vt.T[:, :rank]
        coef = (self.means_ - self.xbar_) @ self.scalings_
        self.intercept_ = -0.5 * xp.sum(coef ** 2, axis=1) + xp.log(self.priors_)
        self.coef_ = coef @ self.scalings_.T
        self.intercept_ -= self.xbar_ @ self.coef_.T

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y):
        if False:
            while True:
                i = 10
        'Fit the Linear Discriminant Analysis model.\n\n           .. versionchanged:: 0.19\n              *store_covariance* has been moved to main constructor.\n\n           .. versionchanged:: 0.19\n              *tol* has been moved to main constructor.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training data.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        Returns\n        -------\n        self : object\n            Fitted estimator.\n        '
        (xp, _) = get_namespace(X)
        (X, y) = self._validate_data(X, y, ensure_min_samples=2, dtype=[xp.float64, xp.float32])
        self.classes_ = unique_labels(y)
        (n_samples, _) = X.shape
        n_classes = self.classes_.shape[0]
        if n_samples == n_classes:
            raise ValueError('The number of samples must be more than the number of classes.')
        if self.priors is None:
            (_, cnts) = xp.unique_counts(y)
            self.priors_ = xp.astype(cnts, X.dtype) / float(y.shape[0])
        else:
            self.priors_ = xp.asarray(self.priors, dtype=X.dtype)
        if xp.any(self.priors_ < 0):
            raise ValueError('priors must be non-negative')
        if xp.abs(xp.sum(self.priors_) - 1.0) > 1e-05:
            warnings.warn('The priors do not sum to 1. Renormalizing', UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()
        max_components = min(n_classes - 1, X.shape[1])
        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError('n_components cannot be larger than min(n_features, n_classes - 1).')
            self._max_components = self.n_components
        if self.solver == 'svd':
            if self.shrinkage is not None:
                raise NotImplementedError("shrinkage not supported with 'svd' solver.")
            if self.covariance_estimator is not None:
                raise ValueError('covariance estimator is not supported with svd solver. Try another solver')
            self._solve_svd(X, y)
        elif self.solver == 'lsqr':
            self._solve_lstsq(X, y, shrinkage=self.shrinkage, covariance_estimator=self.covariance_estimator)
        elif self.solver == 'eigen':
            self._solve_eigen(X, y, shrinkage=self.shrinkage, covariance_estimator=self.covariance_estimator)
        if size(self.classes_) == 2:
            coef_ = xp.asarray(self.coef_[1, :] - self.coef_[0, :], dtype=X.dtype)
            self.coef_ = xp.reshape(coef_, (1, -1))
            intercept_ = xp.asarray(self.intercept_[1] - self.intercept_[0], dtype=X.dtype)
            self.intercept_ = xp.reshape(intercept_, (1,))
        self._n_features_out = self._max_components
        return self

    def transform(self, X):
        if False:
            i = 10
            return i + 15
        "Project data to maximize class separation.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Input data.\n\n        Returns\n        -------\n        X_new : ndarray of shape (n_samples, n_components) or             (n_samples, min(rank, n_components))\n            Transformed data. In the case of the 'svd' solver, the shape\n            is (n_samples, min(rank, n_components)).\n        "
        if self.solver == 'lsqr':
            raise NotImplementedError("transform not implemented for 'lsqr' solver (use 'svd' or 'eigen').")
        check_is_fitted(self)
        (xp, _) = get_namespace(X)
        X = self._validate_data(X, reset=False)
        if self.solver == 'svd':
            X_new = (X - self.xbar_) @ self.scalings_
        elif self.solver == 'eigen':
            X_new = X @ self.scalings_
        return X_new[:, :self._max_components]

    def predict_proba(self, X):
        if False:
            return 10
        'Estimate probability.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Input data.\n\n        Returns\n        -------\n        C : ndarray of shape (n_samples, n_classes)\n            Estimated probabilities.\n        '
        check_is_fitted(self)
        (xp, is_array_api_compliant) = get_namespace(X)
        decision = self.decision_function(X)
        if size(self.classes_) == 2:
            proba = _expit(decision)
            return xp.stack([1 - proba, proba], axis=1)
        else:
            return softmax(decision)

    def predict_log_proba(self, X):
        if False:
            i = 10
            return i + 15
        'Estimate log probability.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Input data.\n\n        Returns\n        -------\n        C : ndarray of shape (n_samples, n_classes)\n            Estimated log probabilities.\n        '
        (xp, _) = get_namespace(X)
        prediction = self.predict_proba(X)
        info = xp.finfo(prediction.dtype)
        if hasattr(info, 'smallest_normal'):
            smallest_normal = info.smallest_normal
        else:
            smallest_normal = info.tiny
        prediction[prediction == 0.0] += smallest_normal
        return xp.log(prediction)

    def decision_function(self, X):
        if False:
            i = 10
            return i + 15
        'Apply decision function to an array of samples.\n\n        The decision function is equal (up to a constant factor) to the\n        log-posterior of the model, i.e. `log p(y = k | x)`. In a binary\n        classification setting this instead corresponds to the difference\n        `log p(y = 1 | x) - log p(y = 0 | x)`. See :ref:`lda_qda_math`.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Array of samples (test vectors).\n\n        Returns\n        -------\n        C : ndarray of shape (n_samples,) or (n_samples, n_classes)\n            Decision function values related to each class, per sample.\n            In the two-class case, the shape is (n_samples,), giving the\n            log likelihood ratio of the positive class.\n        '
        return super().decision_function(X)

    def _more_tags(self):
        if False:
            return 10
        return {'array_api_support': True}

class QuadraticDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    """Quadratic Discriminant Analysis.

    A classifier with a quadratic decision boundary, generated
    by fitting class conditional densities to the data
    and using Bayes' rule.

    The model fits a Gaussian density to each class.

    .. versionadded:: 0.17
       *QuadraticDiscriminantAnalysis*

    Read more in the :ref:`User Guide <lda_qda>`.

    Parameters
    ----------
    priors : array-like of shape (n_classes,), default=None
        Class priors. By default, the class proportions are inferred from the
        training data.

    reg_param : float, default=0.0
        Regularizes the per-class covariance estimates by transforming S2 as
        ``S2 = (1 - reg_param) * S2 + reg_param * np.eye(n_features)``,
        where S2 corresponds to the `scaling_` attribute of a given class.

    store_covariance : bool, default=False
        If True, the class covariance matrices are explicitly computed and
        stored in the `self.covariance_` attribute.

        .. versionadded:: 0.17

    tol : float, default=1.0e-4
        Absolute threshold for a singular value to be considered significant,
        used to estimate the rank of `Xk` where `Xk` is the centered matrix
        of samples in class k. This parameter does not affect the
        predictions. It only controls a warning that is raised when features
        are considered to be colinear.

        .. versionadded:: 0.17

    Attributes
    ----------
    covariance_ : list of len n_classes of ndarray             of shape (n_features, n_features)
        For each class, gives the covariance matrix estimated using the
        samples of that class. The estimations are unbiased. Only present if
        `store_covariance` is True.

    means_ : array-like of shape (n_classes, n_features)
        Class-wise means.

    priors_ : array-like of shape (n_classes,)
        Class priors (sum to 1).

    rotations_ : list of len n_classes of ndarray of shape (n_features, n_k)
        For each class k an array of shape (n_features, n_k), where
        ``n_k = min(n_features, number of elements in class k)``
        It is the rotation of the Gaussian distribution, i.e. its
        principal axis. It corresponds to `V`, the matrix of eigenvectors
        coming from the SVD of `Xk = U S Vt` where `Xk` is the centered
        matrix of samples from class k.

    scalings_ : list of len n_classes of ndarray of shape (n_k,)
        For each class, contains the scaling of
        the Gaussian distributions along its principal axes, i.e. the
        variance in the rotated coordinate system. It corresponds to `S^2 /
        (n_samples - 1)`, where `S` is the diagonal matrix of singular values
        from the SVD of `Xk`, where `Xk` is the centered matrix of samples
        from class k.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    LinearDiscriminantAnalysis : Linear Discriminant Analysis.

    Examples
    --------
    >>> from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = QuadraticDiscriminantAnalysis()
    >>> clf.fit(X, y)
    QuadraticDiscriminantAnalysis()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """
    _parameter_constraints: dict = {'priors': ['array-like', None], 'reg_param': [Interval(Real, 0, 1, closed='both')], 'store_covariance': ['boolean'], 'tol': [Interval(Real, 0, None, closed='left')]}

    def __init__(self, *, priors=None, reg_param=0.0, store_covariance=False, tol=0.0001):
        if False:
            for i in range(10):
                print('nop')
        self.priors = priors
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        if False:
            print('Hello World!')
        'Fit the model according to the given training data and parameters.\n\n            .. versionchanged:: 0.19\n               ``store_covariances`` has been moved to main constructor as\n               ``store_covariance``\n\n            .. versionchanged:: 0.19\n               ``tol`` has been moved to main constructor.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training vector, where `n_samples` is the number of samples and\n            `n_features` is the number of features.\n\n        y : array-like of shape (n_samples,)\n            Target values (integers).\n\n        Returns\n        -------\n        self : object\n            Fitted estimator.\n        '
        (X, y) = self._validate_data(X, y)
        check_classification_targets(y)
        (self.classes_, y) = np.unique(y, return_inverse=True)
        (n_samples, n_features) = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError('The number of classes has to be greater than one; got %d class' % n_classes)
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = np.array(self.priors)
        cov = None
        store_covariance = self.store_covariance
        if store_covariance:
            cov = []
        means = []
        scalings = []
        rotations = []
        for ind in range(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            means.append(meang)
            if len(Xg) == 1:
                raise ValueError('y has only 1 sample in class %s, covariance is ill defined.' % str(self.classes_[ind]))
            Xgc = Xg - meang
            (_, S, Vt) = np.linalg.svd(Xgc, full_matrices=False)
            rank = np.sum(S > self.tol)
            if rank < n_features:
                warnings.warn('Variables are collinear')
            S2 = S ** 2 / (len(Xg) - 1)
            S2 = (1 - self.reg_param) * S2 + self.reg_param
            if self.store_covariance or store_covariance:
                cov.append(np.dot(S2 * Vt.T, Vt))
            scalings.append(S2)
            rotations.append(Vt.T)
        if self.store_covariance or store_covariance:
            self.covariance_ = cov
        self.means_ = np.asarray(means)
        self.scalings_ = scalings
        self.rotations_ = rotations
        return self

    def _decision_function(self, X):
        if False:
            i = 10
            return i + 15
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        norm2 = []
        for i in range(len(self.classes_)):
            R = self.rotations_[i]
            S = self.scalings_[i]
            Xm = X - self.means_[i]
            X2 = np.dot(Xm, R * S ** (-0.5))
            norm2.append(np.sum(X2 ** 2, axis=1))
        norm2 = np.array(norm2).T
        u = np.asarray([np.sum(np.log(s)) for s in self.scalings_])
        return -0.5 * (norm2 + u) + np.log(self.priors_)

    def decision_function(self, X):
        if False:
            i = 10
            return i + 15
        'Apply decision function to an array of samples.\n\n        The decision function is equal (up to a constant factor) to the\n        log-posterior of the model, i.e. `log p(y = k | x)`. In a binary\n        classification setting this instead corresponds to the difference\n        `log p(y = 1 | x) - log p(y = 0 | x)`. See :ref:`lda_qda_math`.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Array of samples (test vectors).\n\n        Returns\n        -------\n        C : ndarray of shape (n_samples,) or (n_samples, n_classes)\n            Decision function values related to each class, per sample.\n            In the two-class case, the shape is (n_samples,), giving the\n            log likelihood ratio of the positive class.\n        '
        dec_func = self._decision_function(X)
        if len(self.classes_) == 2:
            return dec_func[:, 1] - dec_func[:, 0]
        return dec_func

    def predict(self, X):
        if False:
            print('Hello World!')
        'Perform classification on an array of test vectors X.\n\n        The predicted class C for each sample in X is returned.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Vector to be scored, where `n_samples` is the number of samples and\n            `n_features` is the number of features.\n\n        Returns\n        -------\n        C : ndarray of shape (n_samples,)\n            Estimated probabilities.\n        '
        d = self._decision_function(X)
        y_pred = self.classes_.take(d.argmax(1))
        return y_pred

    def predict_proba(self, X):
        if False:
            return 10
        'Return posterior probabilities of classification.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Array of samples/test vectors.\n\n        Returns\n        -------\n        C : ndarray of shape (n_samples, n_classes)\n            Posterior probabilities of classification per class.\n        '
        values = self._decision_function(X)
        likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]

    def predict_log_proba(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Return log of posterior probabilities of classification.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Array of samples/test vectors.\n\n        Returns\n        -------\n        C : ndarray of shape (n_samples, n_classes)\n            Posterior log-probabilities of classification per class.\n        '
        probas_ = self.predict_proba(X)
        return np.log(probas_)