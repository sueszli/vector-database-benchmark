"""
The :mod:`sklearn.pls` module implements Partial Least Squares (PLS).
"""
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.linalg import svd
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, MultiOutputMixin, RegressorMixin, TransformerMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_consistent_length
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import svd_flip
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import FLOAT_DTYPES, check_is_fitted
__all__ = ['PLSCanonical', 'PLSRegression', 'PLSSVD']
if sp_version >= parse_version('1.7'):
    from scipy.linalg import pinv as pinv2
else:
    from scipy.linalg import pinv2

def _pinv2_old(a):
    if False:
        while True:
            i = 10
    (u, s, vh) = svd(a, full_matrices=False, check_finite=False)
    t = u.dtype.char.lower()
    factor = {'f': 1000.0, 'd': 1000000.0}
    cond = np.max(s) * factor[t] * np.finfo(t).eps
    rank = np.sum(s > cond)
    u = u[:, :rank]
    u /= s[:rank]
    return np.transpose(np.conjugate(np.dot(u, vh[:rank])))

def _get_first_singular_vectors_power_method(X, Y, mode='A', max_iter=500, tol=1e-06, norm_y_weights=False):
    if False:
        return 10
    'Return the first left and right singular vectors of X\'Y.\n\n    Provides an alternative to the svd(X\'Y) and uses the power method instead.\n    With norm_y_weights to True and in mode A, this corresponds to the\n    algorithm section 11.3 of the Wegelin\'s review, except this starts at the\n    "update saliences" part.\n    '
    eps = np.finfo(X.dtype).eps
    try:
        y_score = next((col for col in Y.T if np.any(np.abs(col) > eps)))
    except StopIteration as e:
        raise StopIteration('Y residual is constant') from e
    x_weights_old = 100
    if mode == 'B':
        (X_pinv, Y_pinv) = (_pinv2_old(X), _pinv2_old(Y))
    for i in range(max_iter):
        if mode == 'B':
            x_weights = np.dot(X_pinv, y_score)
        else:
            x_weights = np.dot(X.T, y_score) / np.dot(y_score, y_score)
        x_weights /= np.sqrt(np.dot(x_weights, x_weights)) + eps
        x_score = np.dot(X, x_weights)
        if mode == 'B':
            y_weights = np.dot(Y_pinv, x_score)
        else:
            y_weights = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score)
        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights, y_weights)) + eps
        y_score = np.dot(Y, y_weights) / (np.dot(y_weights, y_weights) + eps)
        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        x_weights_old = x_weights
    n_iter = i + 1
    if n_iter == max_iter:
        warnings.warn('Maximum number of iterations reached', ConvergenceWarning)
    return (x_weights, y_weights, n_iter)

def _get_first_singular_vectors_svd(X, Y):
    if False:
        for i in range(10):
            print('nop')
    "Return the first left and right singular vectors of X'Y.\n\n    Here the whole SVD is computed.\n    "
    C = np.dot(X.T, Y)
    (U, _, Vt) = svd(C, full_matrices=False)
    return (U[:, 0], Vt[0, :])

def _center_scale_xy(X, Y, scale=True):
    if False:
        return 10
    'Center X, Y and scale if the scale parameter==True\n\n    Returns\n    -------\n        X, Y, x_mean, y_mean, x_std, y_std\n    '
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return (X, Y, x_mean, y_mean, x_std, y_std)

def _svd_flip_1d(u, v):
    if False:
        for i in range(10):
            print('nop')
    'Same as svd_flip but works on 1d arrays, and is inplace'
    biggest_abs_val_idx = np.argmax(np.abs(u))
    sign = np.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign

class _PLS(ClassNamePrefixFeaturesOutMixin, TransformerMixin, RegressorMixin, MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Partial Least Squares (PLS)

    This class implements the generic PLS algorithm.

    Main ref: Wegelin, a survey of Partial Least Squares (PLS) methods,
    with emphasis on the two-block case
    https://stat.uw.edu/sites/default/files/files/reports/2000/tr371.pdf
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left')], 'scale': ['boolean'], 'deflation_mode': [StrOptions({'regression', 'canonical'})], 'mode': [StrOptions({'A', 'B'})], 'algorithm': [StrOptions({'svd', 'nipals'})], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'tol': [Interval(Real, 0, None, closed='left')], 'copy': ['boolean']}

    @abstractmethod
    def __init__(self, n_components=2, *, scale=True, deflation_mode='regression', mode='A', algorithm='nipals', max_iter=500, tol=1e-06, copy=True):
        if False:
            return 10
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, Y):
        if False:
            return 10
        'Fit model to data.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training vectors, where `n_samples` is the number of samples and\n            `n_features` is the number of predictors.\n\n        Y : array-like of shape (n_samples,) or (n_samples, n_targets)\n            Target vectors, where `n_samples` is the number of samples and\n            `n_targets` is the number of response variables.\n\n        Returns\n        -------\n        self : object\n            Fitted model.\n        '
        check_consistent_length(X, Y)
        X = self._validate_data(X, dtype=np.float64, copy=self.copy, ensure_min_samples=2)
        Y = check_array(Y, input_name='Y', dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            self._predict_1d = True
            Y = Y.reshape(-1, 1)
        else:
            self._predict_1d = False
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]
        n_components = self.n_components
        rank_upper_bound = p if self.deflation_mode == 'regression' else min(n, p, q)
        if n_components > rank_upper_bound:
            raise ValueError(f'`n_components` upper bound is {rank_upper_bound}. Got {n_components} instead. Reduce `n_components`.')
        self._norm_y_weights = self.deflation_mode == 'canonical'
        norm_y_weights = self._norm_y_weights
        (Xk, Yk, self._x_mean, self._y_mean, self._x_std, self._y_std) = _center_scale_xy(X, Y, self.scale)
        self.x_weights_ = np.zeros((p, n_components))
        self.y_weights_ = np.zeros((q, n_components))
        self._x_scores = np.zeros((n, n_components))
        self._y_scores = np.zeros((n, n_components))
        self.x_loadings_ = np.zeros((p, n_components))
        self.y_loadings_ = np.zeros((q, n_components))
        self.n_iter_ = []
        Y_eps = np.finfo(Yk.dtype).eps
        for k in range(n_components):
            if self.algorithm == 'nipals':
                Yk_mask = np.all(np.abs(Yk) < 10 * Y_eps, axis=0)
                Yk[:, Yk_mask] = 0.0
                try:
                    (x_weights, y_weights, n_iter_) = _get_first_singular_vectors_power_method(Xk, Yk, mode=self.mode, max_iter=self.max_iter, tol=self.tol, norm_y_weights=norm_y_weights)
                except StopIteration as e:
                    if str(e) != 'Y residual is constant':
                        raise
                    warnings.warn(f'Y residual is constant at iteration {k}')
                    break
                self.n_iter_.append(n_iter_)
            elif self.algorithm == 'svd':
                (x_weights, y_weights) = _get_first_singular_vectors_svd(Xk, Yk)
            _svd_flip_1d(x_weights, y_weights)
            x_scores = np.dot(Xk, x_weights)
            if norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss
            x_loadings = np.dot(x_scores, Xk) / np.dot(x_scores, x_scores)
            Xk -= np.outer(x_scores, x_loadings)
            if self.deflation_mode == 'canonical':
                y_loadings = np.dot(y_scores, Yk) / np.dot(y_scores, y_scores)
                Yk -= np.outer(y_scores, y_loadings)
            if self.deflation_mode == 'regression':
                y_loadings = np.dot(x_scores, Yk) / np.dot(x_scores, x_scores)
                Yk -= np.outer(x_scores, y_loadings)
            self.x_weights_[:, k] = x_weights
            self.y_weights_[:, k] = y_weights
            self._x_scores[:, k] = x_scores
            self._y_scores[:, k] = y_scores
            self.x_loadings_[:, k] = x_loadings
            self.y_loadings_[:, k] = y_loadings
        self.x_rotations_ = np.dot(self.x_weights_, pinv2(np.dot(self.x_loadings_.T, self.x_weights_), check_finite=False))
        self.y_rotations_ = np.dot(self.y_weights_, pinv2(np.dot(self.y_loadings_.T, self.y_weights_), check_finite=False))
        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.coef_ = (self.coef_ * self._y_std).T
        self.intercept_ = self._y_mean
        self._n_features_out = self.x_rotations_.shape[1]
        return self

    def transform(self, X, Y=None, copy=True):
        if False:
            i = 10
            return i + 15
        'Apply the dimension reduction.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Samples to transform.\n\n        Y : array-like of shape (n_samples, n_targets), default=None\n            Target vectors.\n\n        copy : bool, default=True\n            Whether to copy `X` and `Y`, or perform in-place normalization.\n\n        Returns\n        -------\n        x_scores, y_scores : array-like or tuple of array-like\n            Return `x_scores` if `Y` is not given, `(x_scores, y_scores)` otherwise.\n        '
        check_is_fitted(self)
        X = self._validate_data(X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        X -= self._x_mean
        X /= self._x_std
        x_scores = np.dot(X, self.x_rotations_)
        if Y is not None:
            Y = check_array(Y, input_name='Y', ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Y -= self._y_mean
            Y /= self._y_std
            y_scores = np.dot(Y, self.y_rotations_)
            return (x_scores, y_scores)
        return x_scores

    def inverse_transform(self, X, Y=None):
        if False:
            print('Hello World!')
        'Transform data back to its original space.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_components)\n            New data, where `n_samples` is the number of samples\n            and `n_components` is the number of pls components.\n\n        Y : array-like of shape (n_samples, n_components)\n            New target, where `n_samples` is the number of samples\n            and `n_components` is the number of pls components.\n\n        Returns\n        -------\n        X_reconstructed : ndarray of shape (n_samples, n_features)\n            Return the reconstructed `X` data.\n\n        Y_reconstructed : ndarray of shape (n_samples, n_targets)\n            Return the reconstructed `X` target. Only returned when `Y` is given.\n\n        Notes\n        -----\n        This transformation will only be exact if `n_components=n_features`.\n        '
        check_is_fitted(self)
        X = check_array(X, input_name='X', dtype=FLOAT_DTYPES)
        X_reconstructed = np.matmul(X, self.x_loadings_.T)
        X_reconstructed *= self._x_std
        X_reconstructed += self._x_mean
        if Y is not None:
            Y = check_array(Y, input_name='Y', dtype=FLOAT_DTYPES)
            Y_reconstructed = np.matmul(Y, self.y_loadings_.T)
            Y_reconstructed *= self._y_std
            Y_reconstructed += self._y_mean
            return (X_reconstructed, Y_reconstructed)
        return X_reconstructed

    def predict(self, X, copy=True):
        if False:
            while True:
                i = 10
        'Predict targets of given samples.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Samples.\n\n        copy : bool, default=True\n            Whether to copy `X` and `Y`, or perform in-place normalization.\n\n        Returns\n        -------\n        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)\n            Returns predicted values.\n\n        Notes\n        -----\n        This call requires the estimation of a matrix of shape\n        `(n_features, n_targets)`, which may be an issue in high dimensional\n        space.\n        '
        check_is_fitted(self)
        X = self._validate_data(X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        X -= self._x_mean
        X /= self._x_std
        Ypred = X @ self.coef_.T + self.intercept_
        return Ypred.ravel() if self._predict_1d else Ypred

    def fit_transform(self, X, y=None):
        if False:
            while True:
                i = 10
        'Learn and apply the dimension reduction on the train data.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training vectors, where `n_samples` is the number of samples and\n            `n_features` is the number of predictors.\n\n        y : array-like of shape (n_samples, n_targets), default=None\n            Target vectors, where `n_samples` is the number of samples and\n            `n_targets` is the number of response variables.\n\n        Returns\n        -------\n        self : ndarray of shape (n_samples, n_components)\n            Return `x_scores` if `Y` is not given, `(x_scores, y_scores)` otherwise.\n        '
        return self.fit(X, y).transform(X, y)

    def _more_tags(self):
        if False:
            for i in range(10):
                print('nop')
        return {'poor_score': True, 'requires_y': False}

class PLSRegression(_PLS):
    """PLS regression.

    PLSRegression is also known as PLS2 or PLS1, depending on the number of
    targets.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    max_iter : int, default=500
        The maximum number of iterations of the power method when
        `algorithm='nipals'`. Ignored otherwise.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in :term:`fit` before applying centering,
        and potentially scaling. If `False`, these operations will be done
        inplace, modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_target, n_features)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

        .. versionadded:: 1.1

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> pls2 = PLSRegression(n_components=2)
    >>> pls2.fit(X, Y)
    PLSRegression()
    >>> Y_pred = pls2.predict(X)
    """
    _parameter_constraints: dict = {**_PLS._parameter_constraints}
    for param in ('deflation_mode', 'mode', 'algorithm'):
        _parameter_constraints.pop(param)

    def __init__(self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True):
        if False:
            i = 10
            return i + 15
        super().__init__(n_components=n_components, scale=scale, deflation_mode='regression', mode='A', algorithm='nipals', max_iter=max_iter, tol=tol, copy=copy)

    def fit(self, X, Y):
        if False:
            while True:
                i = 10
        'Fit model to data.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training vectors, where `n_samples` is the number of samples and\n            `n_features` is the number of predictors.\n\n        Y : array-like of shape (n_samples,) or (n_samples, n_targets)\n            Target vectors, where `n_samples` is the number of samples and\n            `n_targets` is the number of response variables.\n\n        Returns\n        -------\n        self : object\n            Fitted model.\n        '
        super().fit(X, Y)
        self.x_scores_ = self._x_scores
        self.y_scores_ = self._y_scores
        return self

class PLSCanonical(_PLS):
    """Partial Least Squares transformer and regressor.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    algorithm : {'nipals', 'svd'}, default='nipals'
        The algorithm used to estimate the first singular vectors of the
        cross-covariance matrix. 'nipals' uses the power method while 'svd'
        will compute the whole SVD.

    max_iter : int, default=500
        The maximum number of iterations of the power method when
        `algorithm='nipals'`. Ignored otherwise.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_targets, n_features)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

        .. versionadded:: 1.1

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component. Empty if `algorithm='svd'`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    CCA : Canonical Correlation Analysis.
    PLSSVD : Partial Least Square SVD.

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSCanonical
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> plsca = PLSCanonical(n_components=2)
    >>> plsca.fit(X, Y)
    PLSCanonical()
    >>> X_c, Y_c = plsca.transform(X, Y)
    """
    _parameter_constraints: dict = {**_PLS._parameter_constraints}
    for param in ('deflation_mode', 'mode'):
        _parameter_constraints.pop(param)

    def __init__(self, n_components=2, *, scale=True, algorithm='nipals', max_iter=500, tol=1e-06, copy=True):
        if False:
            while True:
                i = 10
        super().__init__(n_components=n_components, scale=scale, deflation_mode='canonical', mode='A', algorithm=algorithm, max_iter=max_iter, tol=tol, copy=copy)

class CCA(_PLS):
    """Canonical Correlation Analysis, also known as "Mode B" PLS.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    max_iter : int, default=500
        The maximum number of iterations of the power method.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_targets, n_features)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

        .. versionadded:: 1.1

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.
    PLSSVD : Partial Least Square SVD.

    Examples
    --------
    >>> from sklearn.cross_decomposition import CCA
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> cca = CCA(n_components=1)
    >>> cca.fit(X, Y)
    CCA(n_components=1)
    >>> X_c, Y_c = cca.transform(X, Y)
    """
    _parameter_constraints: dict = {**_PLS._parameter_constraints}
    for param in ('deflation_mode', 'mode', 'algorithm'):
        _parameter_constraints.pop(param)

    def __init__(self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True):
        if False:
            while True:
                i = 10
        super().__init__(n_components=n_components, scale=scale, deflation_mode='canonical', mode='B', algorithm='nipals', max_iter=max_iter, tol=tol, copy=copy)

class PLSSVD(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Partial Least Square SVD.

    This transformer simply performs a SVD on the cross-covariance matrix
    `X'Y`. It is able to project both the training data `X` and the targets
    `Y`. The training data `X` is projected on the left singular vectors, while
    the targets are projected on the right singular vectors.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        The number of components to keep. Should be in `[1,
        min(n_samples, n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If `False`, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in :meth:`transform`.

    y_weights_ : ndarray of (n_targets, n_components)
        The right singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in :meth:`transform`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.
    CCA : Canonical Correlation Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cross_decomposition import PLSSVD
    >>> X = np.array([[0., 0., 1.],
    ...               [1., 0., 0.],
    ...               [2., 2., 2.],
    ...               [2., 5., 4.]])
    >>> Y = np.array([[0.1, -0.2],
    ...               [0.9, 1.1],
    ...               [6.2, 5.9],
    ...               [11.9, 12.3]])
    >>> pls = PLSSVD(n_components=2).fit(X, Y)
    >>> X_c, Y_c = pls.transform(X, Y)
    >>> X_c.shape, Y_c.shape
    ((4, 2), (4, 2))
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left')], 'scale': ['boolean'], 'copy': ['boolean']}

    def __init__(self, n_components=2, *, scale=True, copy=True):
        if False:
            return 10
        self.n_components = n_components
        self.scale = scale
        self.copy = copy

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, Y):
        if False:
            while True:
                i = 10
        'Fit model to data.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training samples.\n\n        Y : array-like of shape (n_samples,) or (n_samples, n_targets)\n            Targets.\n\n        Returns\n        -------\n        self : object\n            Fitted estimator.\n        '
        check_consistent_length(X, Y)
        X = self._validate_data(X, dtype=np.float64, copy=self.copy, ensure_min_samples=2)
        Y = check_array(Y, input_name='Y', dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n_components = self.n_components
        rank_upper_bound = min(X.shape[0], X.shape[1], Y.shape[1])
        if n_components > rank_upper_bound:
            raise ValueError(f'`n_components` upper bound is {rank_upper_bound}. Got {n_components} instead. Reduce `n_components`.')
        (X, Y, self._x_mean, self._y_mean, self._x_std, self._y_std) = _center_scale_xy(X, Y, self.scale)
        C = np.dot(X.T, Y)
        (U, s, Vt) = svd(C, full_matrices=False)
        U = U[:, :n_components]
        Vt = Vt[:n_components]
        (U, Vt) = svd_flip(U, Vt)
        V = Vt.T
        self.x_weights_ = U
        self.y_weights_ = V
        self._n_features_out = self.x_weights_.shape[1]
        return self

    def transform(self, X, Y=None):
        if False:
            return 10
        '\n        Apply the dimensionality reduction.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Samples to be transformed.\n\n        Y : array-like of shape (n_samples,) or (n_samples, n_targets),                 default=None\n            Targets.\n\n        Returns\n        -------\n        x_scores : array-like or tuple of array-like\n            The transformed data `X_transformed` if `Y is not None`,\n            `(X_transformed, Y_transformed)` otherwise.\n        '
        check_is_fitted(self)
        X = self._validate_data(X, dtype=np.float64, reset=False)
        Xr = (X - self._x_mean) / self._x_std
        x_scores = np.dot(Xr, self.x_weights_)
        if Y is not None:
            Y = check_array(Y, input_name='Y', ensure_2d=False, dtype=np.float64)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Yr = (Y - self._y_mean) / self._y_std
            y_scores = np.dot(Yr, self.y_weights_)
            return (x_scores, y_scores)
        return x_scores

    def fit_transform(self, X, y=None):
        if False:
            i = 10
            return i + 15
        'Learn and apply the dimensionality reduction.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training samples.\n\n        y : array-like of shape (n_samples,) or (n_samples, n_targets),                 default=None\n            Targets.\n\n        Returns\n        -------\n        out : array-like or tuple of array-like\n            The transformed data `X_transformed` if `Y is not None`,\n            `(X_transformed, Y_transformed)` otherwise.\n        '
        return self.fit(X, y).transform(X, y)