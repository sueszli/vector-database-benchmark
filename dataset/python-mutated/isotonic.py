import math
import warnings
from numbers import Real
import numpy as np
from scipy import interpolate
from scipy.stats import spearmanr
from ._isotonic import _inplace_contiguous_isotonic_regression, _make_unique
from .base import BaseEstimator, RegressorMixin, TransformerMixin, _fit_context
from .utils import check_array, check_consistent_length
from .utils._param_validation import Interval, StrOptions, validate_params
from .utils.validation import _check_sample_weight, check_is_fitted
__all__ = ['check_increasing', 'isotonic_regression', 'IsotonicRegression']

@validate_params({'x': ['array-like'], 'y': ['array-like']}, prefer_skip_nested_validation=True)
def check_increasing(x, y):
    if False:
        print('Hello World!')
    'Determine whether y is monotonically correlated with x.\n\n    y is found increasing or decreasing with respect to x based on a Spearman\n    correlation test.\n\n    Parameters\n    ----------\n    x : array-like of shape (n_samples,)\n            Training data.\n\n    y : array-like of shape (n_samples,)\n        Training target.\n\n    Returns\n    -------\n    increasing_bool : boolean\n        Whether the relationship is increasing or decreasing.\n\n    Notes\n    -----\n    The Spearman correlation coefficient is estimated from the data, and the\n    sign of the resulting estimate is used as the result.\n\n    In the event that the 95% confidence interval based on Fisher transform\n    spans zero, a warning is raised.\n\n    References\n    ----------\n    Fisher transformation. Wikipedia.\n    https://en.wikipedia.org/wiki/Fisher_transformation\n    '
    (rho, _) = spearmanr(x, y)
    increasing_bool = rho >= 0
    if rho not in [-1.0, 1.0] and len(x) > 3:
        F = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
        F_se = 1 / math.sqrt(len(x) - 3)
        rho_0 = math.tanh(F - 1.96 * F_se)
        rho_1 = math.tanh(F + 1.96 * F_se)
        if np.sign(rho_0) != np.sign(rho_1):
            warnings.warn('Confidence interval of the Spearman correlation coefficient spans zero. Determination of ``increasing`` may be suspect.')
    return increasing_bool

@validate_params({'y': ['array-like'], 'sample_weight': ['array-like', None], 'y_min': [Interval(Real, None, None, closed='both'), None], 'y_max': [Interval(Real, None, None, closed='both'), None], 'increasing': ['boolean']}, prefer_skip_nested_validation=True)
def isotonic_regression(y, *, sample_weight=None, y_min=None, y_max=None, increasing=True):
    if False:
        return 10
    'Solve the isotonic regression model.\n\n    Read more in the :ref:`User Guide <isotonic>`.\n\n    Parameters\n    ----------\n    y : array-like of shape (n_samples,)\n        The data.\n\n    sample_weight : array-like of shape (n_samples,), default=None\n        Weights on each point of the regression.\n        If None, weight is set to 1 (equal weights).\n\n    y_min : float, default=None\n        Lower bound on the lowest predicted value (the minimum value may\n        still be higher). If not set, defaults to -inf.\n\n    y_max : float, default=None\n        Upper bound on the highest predicted value (the maximum may still be\n        lower). If not set, defaults to +inf.\n\n    increasing : bool, default=True\n        Whether to compute ``y_`` is increasing (if set to True) or decreasing\n        (if set to False).\n\n    Returns\n    -------\n    y_ : ndarray of shape (n_samples,)\n        Isotonic fit of y.\n\n    References\n    ----------\n    "Active set algorithms for isotonic regression; A unifying framework"\n    by Michael J. Best and Nilotpal Chakravarti, section 3.\n    '
    order = np.s_[:] if increasing else np.s_[::-1]
    y = check_array(y, ensure_2d=False, input_name='y', dtype=[np.float64, np.float32])
    y = np.array(y[order], dtype=y.dtype)
    sample_weight = _check_sample_weight(sample_weight, y, dtype=y.dtype, copy=True)
    sample_weight = np.ascontiguousarray(sample_weight[order])
    _inplace_contiguous_isotonic_regression(y, sample_weight)
    if y_min is not None or y_max is not None:
        if y_min is None:
            y_min = -np.inf
        if y_max is None:
            y_max = np.inf
        np.clip(y, y_min, y_max, y)
    return y[order]

class IsotonicRegression(RegressorMixin, TransformerMixin, BaseEstimator):
    """Isotonic regression model.

    Read more in the :ref:`User Guide <isotonic>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    y_min : float, default=None
        Lower bound on the lowest predicted value (the minimum value may
        still be higher). If not set, defaults to -inf.

    y_max : float, default=None
        Upper bound on the highest predicted value (the maximum may still be
        lower). If not set, defaults to +inf.

    increasing : bool or 'auto', default=True
        Determines whether the predictions should be constrained to increase
        or decrease with `X`. 'auto' will decide based on the Spearman
        correlation estimate's sign.

    out_of_bounds : {'nan', 'clip', 'raise'}, default='nan'
        Handles how `X` values outside of the training domain are handled
        during prediction.

        - 'nan', predictions will be NaN.
        - 'clip', predictions will be set to the value corresponding to
          the nearest train interval endpoint.
        - 'raise', a `ValueError` is raised.

    Attributes
    ----------
    X_min_ : float
        Minimum value of input array `X_` for left bound.

    X_max_ : float
        Maximum value of input array `X_` for right bound.

    X_thresholds_ : ndarray of shape (n_thresholds,)
        Unique ascending `X` values used to interpolate
        the y = f(X) monotonic function.

        .. versionadded:: 0.24

    y_thresholds_ : ndarray of shape (n_thresholds,)
        De-duplicated `y` values suitable to interpolate the y = f(X)
        monotonic function.

        .. versionadded:: 0.24

    f_ : function
        The stepwise interpolating function that covers the input domain ``X``.

    increasing_ : bool
        Inferred value for ``increasing``.

    See Also
    --------
    sklearn.linear_model.LinearRegression : Ordinary least squares Linear
        Regression.
    sklearn.ensemble.HistGradientBoostingRegressor : Gradient boosting that
        is a non-parametric model accepting monotonicity constraints.
    isotonic_regression : Function to solve the isotonic regression model.

    Notes
    -----
    Ties are broken using the secondary method from de Leeuw, 1977.

    References
    ----------
    Isotonic Median Regression: A Linear Programming Approach
    Nilotpal Chakravarti
    Mathematics of Operations Research
    Vol. 14, No. 2 (May, 1989), pp. 303-308

    Isotone Optimization in R : Pool-Adjacent-Violators
    Algorithm (PAVA) and Active Set Methods
    de Leeuw, Hornik, Mair
    Journal of Statistical Software 2009

    Correctness of Kruskal's algorithms for monotone regression with ties
    de Leeuw, Psychometrica, 1977

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.isotonic import IsotonicRegression
    >>> X, y = make_regression(n_samples=10, n_features=1, random_state=41)
    >>> iso_reg = IsotonicRegression().fit(X, y)
    >>> iso_reg.predict([.1, .2])
    array([1.8628..., 3.7256...])
    """
    _parameter_constraints: dict = {'y_min': [Interval(Real, None, None, closed='both'), None], 'y_max': [Interval(Real, None, None, closed='both'), None], 'increasing': ['boolean', StrOptions({'auto'})], 'out_of_bounds': [StrOptions({'nan', 'clip', 'raise'})]}

    def __init__(self, *, y_min=None, y_max=None, increasing=True, out_of_bounds='nan'):
        if False:
            i = 10
            return i + 15
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

    def _check_input_data_shape(self, X):
        if False:
            print('Hello World!')
        if not (X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1)):
            msg = 'Isotonic regression input X should be a 1d array or 2d array with 1 feature'
            raise ValueError(msg)

    def _build_f(self, X, y):
        if False:
            return 10
        'Build the f_ interp1d function.'
        bounds_error = self.out_of_bounds == 'raise'
        if len(y) == 1:
            self.f_ = lambda x: y.repeat(x.shape)
        else:
            self.f_ = interpolate.interp1d(X, y, kind='linear', bounds_error=bounds_error)

    def _build_y(self, X, y, sample_weight, trim_duplicates=True):
        if False:
            for i in range(10):
                print('nop')
        'Build the y_ IsotonicRegression.'
        self._check_input_data_shape(X)
        X = X.reshape(-1)
        if self.increasing == 'auto':
            self.increasing_ = check_increasing(X, y)
        else:
            self.increasing_ = self.increasing
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        mask = sample_weight > 0
        (X, y, sample_weight) = (X[mask], y[mask], sample_weight[mask])
        order = np.lexsort((y, X))
        (X, y, sample_weight) = [array[order] for array in [X, y, sample_weight]]
        (unique_X, unique_y, unique_sample_weight) = _make_unique(X, y, sample_weight)
        X = unique_X
        y = isotonic_regression(unique_y, sample_weight=unique_sample_weight, y_min=self.y_min, y_max=self.y_max, increasing=self.increasing_)
        (self.X_min_, self.X_max_) = (np.min(X), np.max(X))
        if trim_duplicates:
            keep_data = np.ones((len(y),), dtype=bool)
            keep_data[1:-1] = np.logical_or(np.not_equal(y[1:-1], y[:-2]), np.not_equal(y[1:-1], y[2:]))
            return (X[keep_data], y[keep_data])
        else:
            return (X, y)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        if False:
            while True:
                i = 10
        'Fit the model using X, y as training data.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples,) or (n_samples, 1)\n            Training data.\n\n            .. versionchanged:: 0.24\n               Also accepts 2d array with 1 feature.\n\n        y : array-like of shape (n_samples,)\n            Training target.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Weights. If set to None, all weights will be set to 1 (equal\n            weights).\n\n        Returns\n        -------\n        self : object\n            Returns an instance of self.\n\n        Notes\n        -----\n        X is stored for future use, as :meth:`transform` needs X to interpolate\n        new input data.\n        '
        check_params = dict(accept_sparse=False, ensure_2d=False)
        X = check_array(X, input_name='X', dtype=[np.float64, np.float32], **check_params)
        y = check_array(y, input_name='y', dtype=X.dtype, **check_params)
        check_consistent_length(X, y, sample_weight)
        (X, y) = self._build_y(X, y, sample_weight)
        (self.X_thresholds_, self.y_thresholds_) = (X, y)
        self._build_f(X, y)
        return self

    def _transform(self, T):
        if False:
            print('Hello World!')
        '`_transform` is called by both `transform` and `predict` methods.\n\n        Since `transform` is wrapped to output arrays of specific types (e.g.\n        NumPy arrays, pandas DataFrame), we cannot make `predict` call `transform`\n        directly.\n\n        The above behaviour could be changed in the future, if we decide to output\n        other type of arrays when calling `predict`.\n        '
        if hasattr(self, 'X_thresholds_'):
            dtype = self.X_thresholds_.dtype
        else:
            dtype = np.float64
        T = check_array(T, dtype=dtype, ensure_2d=False)
        self._check_input_data_shape(T)
        T = T.reshape(-1)
        if self.out_of_bounds == 'clip':
            T = np.clip(T, self.X_min_, self.X_max_)
        res = self.f_(T)
        res = res.astype(T.dtype)
        return res

    def transform(self, T):
        if False:
            return 10
        'Transform new data by linear interpolation.\n\n        Parameters\n        ----------\n        T : array-like of shape (n_samples,) or (n_samples, 1)\n            Data to transform.\n\n            .. versionchanged:: 0.24\n               Also accepts 2d array with 1 feature.\n\n        Returns\n        -------\n        y_pred : ndarray of shape (n_samples,)\n            The transformed data.\n        '
        return self._transform(T)

    def predict(self, T):
        if False:
            print('Hello World!')
        'Predict new data by linear interpolation.\n\n        Parameters\n        ----------\n        T : array-like of shape (n_samples,) or (n_samples, 1)\n            Data to transform.\n\n        Returns\n        -------\n        y_pred : ndarray of shape (n_samples,)\n            Transformed data.\n        '
        return self._transform(T)

    def get_feature_names_out(self, input_features=None):
        if False:
            while True:
                i = 10
        'Get output feature names for transformation.\n\n        Parameters\n        ----------\n        input_features : array-like of str or None, default=None\n            Ignored.\n\n        Returns\n        -------\n        feature_names_out : ndarray of str objects\n            An ndarray with one string i.e. ["isotonicregression0"].\n        '
        check_is_fitted(self, 'f_')
        class_name = self.__class__.__name__.lower()
        return np.asarray([f'{class_name}0'], dtype=object)

    def __getstate__(self):
        if False:
            print('Hello World!')
        'Pickle-protocol - return state of the estimator.'
        state = super().__getstate__()
        state.pop('f_', None)
        return state

    def __setstate__(self, state):
        if False:
            return 10
        'Pickle-protocol - set state of the estimator.\n\n        We need to rebuild the interpolation function.\n        '
        super().__setstate__(state)
        if hasattr(self, 'X_thresholds_') and hasattr(self, 'y_thresholds_'):
            self._build_f(self.X_thresholds_, self.y_thresholds_)

    def _more_tags(self):
        if False:
            while True:
                i = 10
        return {'X_types': ['1darray']}