"""
Hannan-Rissanen procedure for estimating ARMA(p,q) model parameters.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
from scipy.signal import lfilter
from statsmodels.tools.tools import Bunch
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams

def hannan_rissanen(endog, ar_order=0, ma_order=0, demean=True, initial_ar_order=None, unbiased=None, fixed_params=None):
    if False:
        while True:
            i = 10
    '\n    Estimate ARMA parameters using Hannan-Rissanen procedure.\n\n    Parameters\n    ----------\n    endog : array_like\n        Input time series array, assumed to be stationary.\n    ar_order : int or list of int\n        Autoregressive order\n    ma_order : int or list of int\n        Moving average order\n    demean : bool, optional\n        Whether to estimate and remove the mean from the process prior to\n        fitting the ARMA coefficients. Default is True.\n    initial_ar_order : int, optional\n        Order of long autoregressive process used for initial computation of\n        residuals.\n    unbiased : bool, optional\n        Whether or not to apply the bias correction step. Default is True if\n        the estimated coefficients from the previous step imply a stationary\n        and invertible process and False otherwise.\n    fixed_params : dict, optional\n        Dictionary with names of fixed parameters as keys (e.g. \'ar.L1\',\n        \'ma.L2\'), which correspond to SARIMAXSpecification.param_names.\n        Dictionary values are the values of the associated fixed parameters.\n\n    Returns\n    -------\n    parameters : SARIMAXParams object\n    other_results : Bunch\n        Includes three components: `spec`, containing the\n        `SARIMAXSpecification` instance corresponding to the input arguments;\n        `initial_ar_order`, containing the autoregressive lag order used in the\n        first step; and `resid`, which contains the computed residuals from the\n        last step.\n\n    Notes\n    -----\n    The primary reference is [1]_, section 5.1.4, which describes a three-step\n    procedure that we implement here.\n\n    1. Fit a large-order AR model via Yule-Walker to estimate residuals\n    2. Compute AR and MA estimates via least squares\n    3. (Unless the estimated coefficients from step (2) are non-stationary /\n       non-invertible or `unbiased=False`) Perform bias correction\n\n    The order used for the AR model in the first step may be given as an\n    argument. If it is not, we compute it as suggested by [2]_.\n\n    The estimate of the variance that we use is computed from the residuals\n    of the least-squares regression and not from the innovations algorithm.\n    This is because our fast implementation of the innovations algorithm is\n    only valid for stationary processes, and the Hannan-Rissanen procedure may\n    produce estimates that imply non-stationary processes. To avoid\n    inconsistency, we never compute this latter variance here, even if it is\n    possible. See test_hannan_rissanen::test_brockwell_davis_example_517 for\n    an example of how to compute this variance manually.\n\n    This procedure assumes that the series is stationary, but if this is not\n    true, it is still possible that this procedure will return parameters that\n    imply a non-stationary / non-invertible process.\n\n    Note that the third stage will only be applied if the parameters from the\n    second stage imply a stationary / invertible model. If `unbiased=True` is\n    given, then non-stationary / non-invertible parameters in the second stage\n    will throw an exception.\n\n    References\n    ----------\n    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.\n       Introduction to Time Series and Forecasting. Springer.\n    .. [2] Gomez, Victor, and Agustin Maravall. 2001.\n       "Automatic Modeling Methods for Univariate Series."\n       A Course in Time Series Analysis, 171–201.\n    '
    spec = SARIMAXSpecification(endog, ar_order=ar_order, ma_order=ma_order)
    fixed_params = _validate_fixed_params(fixed_params, spec.param_names)
    endog = spec.endog
    if demean:
        endog = endog - endog.mean()
    p = SARIMAXParams(spec=spec)
    nobs = len(endog)
    max_ar_order = spec.max_ar_order
    max_ma_order = spec.max_ma_order
    if initial_ar_order is None:
        initial_ar_order = max(np.floor(np.log(nobs) ** 2).astype(int), 2 * max(max_ar_order, max_ma_order))
    _ = SARIMAXSpecification(endog, ar_order=initial_ar_order)
    params_info = _package_fixed_and_free_params_info(fixed_params, spec.ar_lags, spec.ma_lags)
    lagged_endog = lagmat(endog, max_ar_order, trim='both')
    mod = None
    if max_ma_order == 0 and max_ar_order == 0:
        p.sigma2 = np.var(endog, ddof=0)
        resid = endog.copy()
    elif max_ma_order == 0:
        X_with_free_params = lagged_endog[:, params_info.free_ar_ix]
        X_with_fixed_params = lagged_endog[:, params_info.fixed_ar_ix]
        y = endog[max_ar_order:]
        if X_with_fixed_params.shape[1] != 0:
            y = y - X_with_fixed_params.dot(params_info.fixed_ar_params)
        if X_with_free_params.shape[1] == 0:
            p.ar_params = params_info.fixed_ar_params
            p.sigma2 = np.var(y, ddof=0)
            resid = y.copy()
        else:
            mod = OLS(y, X_with_free_params)
            res = mod.fit()
            resid = res.resid
            p.sigma2 = res.scale
            p.ar_params = _stitch_fixed_and_free_params(fixed_ar_or_ma_lags=params_info.fixed_ar_lags, fixed_ar_or_ma_params=params_info.fixed_ar_params, free_ar_or_ma_lags=params_info.free_ar_lags, free_ar_or_ma_params=res.params, spec_ar_or_ma_lags=spec.ar_lags)
    else:
        (initial_ar_params, _) = yule_walker(endog, order=initial_ar_order, method='mle')
        X = lagmat(endog, initial_ar_order, trim='both')
        y = endog[initial_ar_order:]
        resid = y - X.dot(initial_ar_params)
        lagged_resid = lagmat(resid, max_ma_order, trim='both')
        ix = initial_ar_order + max_ma_order - max_ar_order
        X_with_free_params = np.c_[lagged_endog[ix:, params_info.free_ar_ix], lagged_resid[:, params_info.free_ma_ix]]
        X_with_fixed_params = np.c_[lagged_endog[ix:, params_info.fixed_ar_ix], lagged_resid[:, params_info.fixed_ma_ix]]
        y = endog[initial_ar_order + max_ma_order:]
        if X_with_fixed_params.shape[1] != 0:
            y = y - X_with_fixed_params.dot(np.r_[params_info.fixed_ar_params, params_info.fixed_ma_params])
        if X_with_free_params.shape[1] == 0:
            p.ar_params = params_info.fixed_ar_params
            p.ma_params = params_info.fixed_ma_params
            p.sigma2 = np.var(y, ddof=0)
            resid = y.copy()
        else:
            mod = OLS(y, X_with_free_params)
            res = mod.fit()
            k_free_ar_params = len(params_info.free_ar_lags)
            p.ar_params = _stitch_fixed_and_free_params(fixed_ar_or_ma_lags=params_info.fixed_ar_lags, fixed_ar_or_ma_params=params_info.fixed_ar_params, free_ar_or_ma_lags=params_info.free_ar_lags, free_ar_or_ma_params=res.params[:k_free_ar_params], spec_ar_or_ma_lags=spec.ar_lags)
            p.ma_params = _stitch_fixed_and_free_params(fixed_ar_or_ma_lags=params_info.fixed_ma_lags, fixed_ar_or_ma_params=params_info.fixed_ma_params, free_ar_or_ma_lags=params_info.free_ma_lags, free_ar_or_ma_params=res.params[k_free_ar_params:], spec_ar_or_ma_lags=spec.ma_lags)
            resid = res.resid
            p.sigma2 = res.scale
        if unbiased is True:
            if len(fixed_params) != 0:
                raise NotImplementedError('Third step of Hannan-Rissanen estimation to remove parameter bias is not yet implemented for the case with fixed parameters.')
            elif not (p.is_stationary and p.is_invertible):
                raise ValueError('Cannot perform third step of Hannan-Rissanen estimation to remove parameter bias, because parameters estimated from the second step are non-stationary or non-invertible.')
        elif unbiased is None:
            if len(fixed_params) != 0:
                unbiased = False
            else:
                unbiased = p.is_stationary and p.is_invertible
        if unbiased is True:
            if mod is None:
                raise ValueError('Must have free parameters to use unbiased')
            Z = np.zeros_like(endog)
            ar_coef = p.ar_poly.coef
            ma_coef = p.ma_poly.coef
            for t in range(nobs):
                if t >= max(max_ar_order, max_ma_order):
                    tmp_ar = np.dot(-ar_coef[1:], endog[t - max_ar_order:t][::-1])
                    tmp_ma = np.dot(ma_coef[1:], Z[t - max_ma_order:t][::-1])
                    Z[t] = endog[t] - tmp_ar - tmp_ma
            V = lfilter([1], ar_coef, Z)
            W = lfilter(np.r_[1, -ma_coef[1:]], [1], Z)
            lagged_V = lagmat(V, max_ar_order, trim='both')
            lagged_W = lagmat(W, max_ma_order, trim='both')
            exog = np.c_[lagged_V[max(max_ma_order - max_ar_order, 0):, params_info.free_ar_ix], lagged_W[max(max_ar_order - max_ma_order, 0):, params_info.free_ma_ix]]
            mod_unbias = OLS(Z[max(max_ar_order, max_ma_order):], exog)
            res_unbias = mod_unbias.fit()
            p.ar_params = p.ar_params + res_unbias.params[:spec.k_ar_params]
            p.ma_params = p.ma_params + res_unbias.params[spec.k_ar_params:]
            resid = mod.endog - mod.exog.dot(np.r_[p.ar_params, p.ma_params])
            p.sigma2 = np.inner(resid, resid) / len(resid)
    other_results = Bunch({'spec': spec, 'initial_ar_order': initial_ar_order, 'resid': resid})
    return (p, other_results)

def _validate_fixed_params(fixed_params, spec_param_names):
    if False:
        while True:
            i = 10
    '\n    Check that keys in fixed_params are a subset of spec.param_names except\n    "sigma2"\n\n    Parameters\n    ----------\n    fixed_params : dict\n    spec_param_names : list of string\n        SARIMAXSpecification.param_names\n    '
    if fixed_params is None:
        fixed_params = {}
    assert isinstance(fixed_params, dict)
    fixed_param_names = set(fixed_params.keys())
    valid_param_names = set(spec_param_names) - {'sigma2'}
    invalid_param_names = fixed_param_names - valid_param_names
    if len(invalid_param_names) > 0:
        raise ValueError(f'Invalid fixed parameter(s): {sorted(list(invalid_param_names))}. Please select among {sorted(list(valid_param_names))}.')
    return fixed_params

def _package_fixed_and_free_params_info(fixed_params, spec_ar_lags, spec_ma_lags):
    if False:
        return 10
    '\n    Parameters\n    ----------\n    fixed_params : dict\n    spec_ar_lags : list of int\n        SARIMAXSpecification.ar_lags\n    spec_ma_lags : list of int\n        SARIMAXSpecification.ma_lags\n\n    Returns\n    -------\n    Bunch with\n    (lags) fixed_ar_lags, fixed_ma_lags, free_ar_lags, free_ma_lags;\n    (ix) fixed_ar_ix, fixed_ma_ix, free_ar_ix, free_ma_ix;\n    (params) fixed_ar_params, free_ma_params\n    '
    fixed_ar_lags_and_params = []
    fixed_ma_lags_and_params = []
    for (key, val) in fixed_params.items():
        lag = int(key.split('.')[-1].lstrip('L'))
        if key.startswith('ar'):
            fixed_ar_lags_and_params.append((lag, val))
        elif key.startswith('ma'):
            fixed_ma_lags_and_params.append((lag, val))
    fixed_ar_lags_and_params.sort()
    fixed_ma_lags_and_params.sort()
    fixed_ar_lags = [lag for (lag, _) in fixed_ar_lags_and_params]
    fixed_ar_params = np.array([val for (_, val) in fixed_ar_lags_and_params])
    fixed_ma_lags = [lag for (lag, _) in fixed_ma_lags_and_params]
    fixed_ma_params = np.array([val for (_, val) in fixed_ma_lags_and_params])
    free_ar_lags = [lag for lag in spec_ar_lags if lag not in set(fixed_ar_lags)]
    free_ma_lags = [lag for lag in spec_ma_lags if lag not in set(fixed_ma_lags)]
    free_ar_ix = np.array(free_ar_lags, dtype=int) - 1
    free_ma_ix = np.array(free_ma_lags, dtype=int) - 1
    fixed_ar_ix = np.array(fixed_ar_lags, dtype=int) - 1
    fixed_ma_ix = np.array(fixed_ma_lags, dtype=int) - 1
    return Bunch(fixed_ar_lags=fixed_ar_lags, fixed_ma_lags=fixed_ma_lags, free_ar_lags=free_ar_lags, free_ma_lags=free_ma_lags, fixed_ar_ix=fixed_ar_ix, fixed_ma_ix=fixed_ma_ix, free_ar_ix=free_ar_ix, free_ma_ix=free_ma_ix, fixed_ar_params=fixed_ar_params, fixed_ma_params=fixed_ma_params)

def _stitch_fixed_and_free_params(fixed_ar_or_ma_lags, fixed_ar_or_ma_params, free_ar_or_ma_lags, free_ar_or_ma_params, spec_ar_or_ma_lags):
    if False:
        while True:
            i = 10
    '\n    Stitch together fixed and free params, by the order of lags, for setting\n    SARIMAXParams.ma_params or SARIMAXParams.ar_params\n\n    Parameters\n    ----------\n    fixed_ar_or_ma_lags : list or np.array\n    fixed_ar_or_ma_params : list or np.array\n        fixed_ar_or_ma_params corresponds with fixed_ar_or_ma_lags\n    free_ar_or_ma_lags : list or np.array\n    free_ar_or_ma_params : list or np.array\n        free_ar_or_ma_params corresponds with free_ar_or_ma_lags\n    spec_ar_or_ma_lags : list\n        SARIMAXSpecification.ar_lags or SARIMAXSpecification.ma_lags\n\n    Returns\n    -------\n    list of fixed and free params by the order of lags\n    '
    assert len(fixed_ar_or_ma_lags) == len(fixed_ar_or_ma_params)
    assert len(free_ar_or_ma_lags) == len(free_ar_or_ma_params)
    all_lags = np.r_[fixed_ar_or_ma_lags, free_ar_or_ma_lags]
    all_params = np.r_[fixed_ar_or_ma_params, free_ar_or_ma_params]
    assert set(all_lags) == set(spec_ar_or_ma_lags)
    lag_to_param_map = dict(zip(all_lags, all_params))
    all_params_sorted = [lag_to_param_map[lag] for lag in spec_ar_or_ma_lags]
    return all_params_sorted