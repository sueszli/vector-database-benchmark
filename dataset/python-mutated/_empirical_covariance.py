"""
Maximum likelihood covariance estimator.

"""
import warnings
import numpy as np
from scipy import linalg
from .. import config_context
from ..base import BaseEstimator, _fit_context
from ..metrics.pairwise import pairwise_distances
from ..utils import check_array
from ..utils._param_validation import validate_params
from ..utils.extmath import fast_logdet

@validate_params({'emp_cov': [np.ndarray], 'precision': [np.ndarray]}, prefer_skip_nested_validation=True)
def log_likelihood(emp_cov, precision):
    if False:
        for i in range(10):
            print('nop')
    'Compute the sample mean of the log_likelihood under a covariance model.\n\n    Computes the empirical expected log-likelihood, allowing for universal\n    comparison (beyond this software package), and accounts for normalization\n    terms and scaling.\n\n    Parameters\n    ----------\n    emp_cov : ndarray of shape (n_features, n_features)\n        Maximum Likelihood Estimator of covariance.\n\n    precision : ndarray of shape (n_features, n_features)\n        The precision matrix of the covariance model to be tested.\n\n    Returns\n    -------\n    log_likelihood_ : float\n        Sample mean of the log-likelihood.\n    '
    p = precision.shape[0]
    log_likelihood_ = -np.sum(emp_cov * precision) + fast_logdet(precision)
    log_likelihood_ -= p * np.log(2 * np.pi)
    log_likelihood_ /= 2.0
    return log_likelihood_

@validate_params({'X': ['array-like'], 'assume_centered': ['boolean']}, prefer_skip_nested_validation=True)
def empirical_covariance(X, *, assume_centered=False):
    if False:
        print('Hello World!')
    'Compute the Maximum likelihood covariance estimator.\n\n    Parameters\n    ----------\n    X : ndarray of shape (n_samples, n_features)\n        Data from which to compute the covariance estimate.\n\n    assume_centered : bool, default=False\n        If `True`, data will not be centered before computation.\n        Useful when working with data whose mean is almost, but not exactly\n        zero.\n        If `False`, data will be centered before computation.\n\n    Returns\n    -------\n    covariance : ndarray of shape (n_features, n_features)\n        Empirical covariance (Maximum Likelihood Estimator).\n\n    Examples\n    --------\n    >>> from sklearn.covariance import empirical_covariance\n    >>> X = [[1,1,1],[1,1,1],[1,1,1],\n    ...      [0,0,0],[0,0,0],[0,0,0]]\n    >>> empirical_covariance(X)\n    array([[0.25, 0.25, 0.25],\n           [0.25, 0.25, 0.25],\n           [0.25, 0.25, 0.25]])\n    '
    X = check_array(X, ensure_2d=False, force_all_finite=False)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
    if X.shape[0] == 1:
        warnings.warn('Only one sample available. You may want to reshape your data array')
    if assume_centered:
        covariance = np.dot(X.T, X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)
    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance

class EmpiricalCovariance(BaseEstimator):
    """Maximum likelihood covariance estimator.

    Read more in the :ref:`User Guide <covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specifies if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo-inverse matrix.
        (stored only if store_precision is True)

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    EllipticEnvelope : An object for detecting outliers in
        a Gaussian distributed dataset.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    LedoitWolf : LedoitWolf Estimator.
    MinCovDet : Minimum Covariance Determinant
        (robust estimator of covariance).
    OAS : Oracle Approximating Shrinkage Estimator.
    ShrunkCovariance : Covariance estimator with shrinkage.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import EmpiricalCovariance
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> real_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0],
    ...                             cov=real_cov,
    ...                             size=500)
    >>> cov = EmpiricalCovariance().fit(X)
    >>> cov.covariance_
    array([[0.7569..., 0.2818...],
           [0.2818..., 0.3928...]])
    >>> cov.location_
    array([0.0622..., 0.0193...])
    """
    _parameter_constraints: dict = {'store_precision': ['boolean'], 'assume_centered': ['boolean']}

    def __init__(self, *, store_precision=True, assume_centered=False):
        if False:
            return 10
        self.store_precision = store_precision
        self.assume_centered = assume_centered

    def _set_covariance(self, covariance):
        if False:
            while True:
                i = 10
        'Saves the covariance and precision estimates\n\n        Storage is done accordingly to `self.store_precision`.\n        Precision stored only if invertible.\n\n        Parameters\n        ----------\n        covariance : array-like of shape (n_features, n_features)\n            Estimated covariance matrix to be stored, and from which precision\n            is computed.\n        '
        covariance = check_array(covariance)
        self.covariance_ = covariance
        if self.store_precision:
            self.precision_ = linalg.pinvh(covariance, check_finite=False)
        else:
            self.precision_ = None

    def get_precision(self):
        if False:
            i = 10
            return i + 15
        'Getter for the precision matrix.\n\n        Returns\n        -------\n        precision_ : array-like of shape (n_features, n_features)\n            The precision matrix associated to the current covariance object.\n        '
        if self.store_precision:
            precision = self.precision_
        else:
            precision = linalg.pinvh(self.covariance_, check_finite=False)
        return precision

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        if False:
            return 10
        'Fit the maximum likelihood covariance estimator to X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n          Training data, where `n_samples` is the number of samples and\n          `n_features` is the number of features.\n\n        y : Ignored\n            Not used, present for API consistency by convention.\n\n        Returns\n        -------\n        self : object\n            Returns the instance itself.\n        '
        X = self._validate_data(X)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        covariance = empirical_covariance(X, assume_centered=self.assume_centered)
        self._set_covariance(covariance)
        return self

    def score(self, X_test, y=None):
        if False:
            return 10
        'Compute the log-likelihood of `X_test` under the estimated Gaussian model.\n\n        The Gaussian model is defined by its mean and covariance matrix which are\n        represented respectively by `self.location_` and `self.covariance_`.\n\n        Parameters\n        ----------\n        X_test : array-like of shape (n_samples, n_features)\n            Test data of which we compute the likelihood, where `n_samples` is\n            the number of samples and `n_features` is the number of features.\n            `X_test` is assumed to be drawn from the same distribution than\n            the data used in fit (including centering).\n\n        y : Ignored\n            Not used, present for API consistency by convention.\n\n        Returns\n        -------\n        res : float\n            The log-likelihood of `X_test` with `self.location_` and `self.covariance_`\n            as estimators of the Gaussian model mean and covariance matrix respectively.\n        '
        X_test = self._validate_data(X_test, reset=False)
        test_cov = empirical_covariance(X_test - self.location_, assume_centered=True)
        res = log_likelihood(test_cov, self.get_precision())
        return res

    def error_norm(self, comp_cov, norm='frobenius', scaling=True, squared=True):
        if False:
            for i in range(10):
                print('nop')
        'Compute the Mean Squared Error between two covariance estimators.\n\n        Parameters\n        ----------\n        comp_cov : array-like of shape (n_features, n_features)\n            The covariance to compare with.\n\n        norm : {"frobenius", "spectral"}, default="frobenius"\n            The type of norm used to compute the error. Available error types:\n            - \'frobenius\' (default): sqrt(tr(A^t.A))\n            - \'spectral\': sqrt(max(eigenvalues(A^t.A))\n            where A is the error ``(comp_cov - self.covariance_)``.\n\n        scaling : bool, default=True\n            If True (default), the squared error norm is divided by n_features.\n            If False, the squared error norm is not rescaled.\n\n        squared : bool, default=True\n            Whether to compute the squared error norm or the error norm.\n            If True (default), the squared error norm is returned.\n            If False, the error norm is returned.\n\n        Returns\n        -------\n        result : float\n            The Mean Squared Error (in the sense of the Frobenius norm) between\n            `self` and `comp_cov` covariance estimators.\n        '
        error = comp_cov - self.covariance_
        if norm == 'frobenius':
            squared_norm = np.sum(error ** 2)
        elif norm == 'spectral':
            squared_norm = np.amax(linalg.svdvals(np.dot(error.T, error)))
        else:
            raise NotImplementedError('Only spectral and frobenius norms are implemented')
        if scaling:
            squared_norm = squared_norm / error.shape[0]
        if squared:
            result = squared_norm
        else:
            result = np.sqrt(squared_norm)
        return result

    def mahalanobis(self, X):
        if False:
            while True:
                i = 10
        'Compute the squared Mahalanobis distances of given observations.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The observations, the Mahalanobis distances of the which we\n            compute. Observations are assumed to be drawn from the same\n            distribution than the data used in fit.\n\n        Returns\n        -------\n        dist : ndarray of shape (n_samples,)\n            Squared Mahalanobis distances of the observations.\n        '
        X = self._validate_data(X, reset=False)
        precision = self.get_precision()
        with config_context(assume_finite=True):
            dist = pairwise_distances(X, self.location_[np.newaxis, :], metric='mahalanobis', VI=precision)
        return np.reshape(dist, (len(X),)) ** 2