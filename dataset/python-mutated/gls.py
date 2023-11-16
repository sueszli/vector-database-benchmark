"""
Feasible generalized least squares for regression with SARIMA errors.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
import warnings
from statsmodels.tools.tools import add_constant, Bunch
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.innovations import arma_innovations
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import innovations, innovations_mle
from statsmodels.tsa.arima.estimators.statespace import statespace
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams

def gls(endog, exog=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0), include_constant=None, n_iter=None, max_iter=50, tolerance=1e-08, arma_estimator='innovations_mle', arma_estimator_kwargs=None):
    if False:
        print('Hello World!')
    "\n    Estimate ARMAX parameters by GLS.\n\n    Parameters\n    ----------\n    endog : array_like\n        Input time series array.\n    exog : array_like, optional\n        Array of exogenous regressors. If not included, then `include_constant`\n        must be True, and then `exog` will only include the constant column.\n    order : tuple, optional\n        The (p,d,q) order of the ARIMA model. Default is (0, 0, 0).\n    seasonal_order : tuple, optional\n        The (P,D,Q,s) order of the seasonal ARIMA model.\n        Default is (0, 0, 0, 0).\n    include_constant : bool, optional\n        Whether to add a constant term in `exog` if it's not already there.\n        The estimate of the constant will then appear as one of the `exog`\n        parameters. If `exog` is None, then the constant will represent the\n        mean of the process. Default is True if the specified model does not\n        include integration and False otherwise.\n    n_iter : int, optional\n        Optionally iterate feasible GSL a specific number of times. Default is\n        to iterate to convergence. If set, this argument overrides the\n        `max_iter` and `tolerance` arguments.\n    max_iter : int, optional\n        Maximum number of feasible GLS iterations. Default is 50. If `n_iter`\n        is set, it overrides this argument.\n    tolerance : float, optional\n        Tolerance for determining convergence of feasible GSL iterations. If\n        `iter` is set, this argument has no effect.\n        Default is 1e-8.\n    arma_estimator : str, optional\n        The estimator used for estimating the ARMA model. This option should\n        not generally be used, unless the default method is failing or is\n        otherwise unsuitable. Not all values will be valid, depending on the\n        specified model orders (`order` and `seasonal_order`). Possible values\n        are:\n        * 'innovations_mle' - can be used with any specification\n        * 'statespace' - can be used with any specification\n        * 'hannan_rissanen' - can be used with any ARMA non-seasonal model\n        * 'yule_walker' - only non-seasonal consecutive\n          autoregressive (AR) models\n        * 'burg' - only non-seasonal, consecutive autoregressive (AR) models\n        * 'innovations' - only non-seasonal, consecutive moving\n          average (MA) models.\n        The default is 'innovations_mle'.\n    arma_estimator_kwargs : dict, optional\n        Arguments to pass to the ARMA estimator.\n\n    Returns\n    -------\n    parameters : SARIMAXParams object\n        Contains the parameter estimates from the final iteration.\n    other_results : Bunch\n        Includes eight components: `spec`, `params`, `converged`,\n        `differences`, `iterations`, `arma_estimator`, 'arma_estimator_kwargs',\n        and `arma_results`.\n\n    Notes\n    -----\n    The primary reference is [1]_, section 6.6. In particular, the\n    implementation follows the iterative procedure described in section 6.6.2.\n    Construction of the transformed variables used to compute the GLS estimator\n    described in section 6.6.1 is done via an application of the innovations\n    algorithm (rather than explicit construction of the transformation matrix).\n\n    Note that if the specified model includes integration, both the `endog` and\n    `exog` series will be differenced prior to estimation and a warning will\n    be issued to alert the user.\n\n    References\n    ----------\n    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.\n       Introduction to Time Series and Forecasting. Springer.\n    "
    if n_iter is not None:
        max_iter = n_iter
        tolerance = np.inf
    integrated = order[1] > 0 or seasonal_order[1] > 0
    if include_constant is None:
        include_constant = not integrated
    elif include_constant and integrated:
        raise ValueError('Cannot include a constant in an integrated model.')
    if include_constant:
        exog = np.ones_like(endog) if exog is None else add_constant(exog)
    spec = SARIMAXSpecification(endog, exog=exog, order=order, seasonal_order=seasonal_order)
    endog = spec.endog
    exog = spec.exog
    if spec.is_integrated:
        warnings.warn('Provided `endog` and `exog` series have been differenced to eliminate integration prior to GLS parameter estimation.')
        endog = diff(endog, k_diff=spec.diff, k_seasonal_diff=spec.seasonal_diff, seasonal_periods=spec.seasonal_periods)
        exog = diff(exog, k_diff=spec.diff, k_seasonal_diff=spec.seasonal_diff, seasonal_periods=spec.seasonal_periods)
    augmented = np.c_[endog, exog]
    spec.validate_estimator(arma_estimator)
    if arma_estimator_kwargs is None:
        arma_estimator_kwargs = {}
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    exog_params = res_ols.params
    resid = res_ols.resid
    p = SARIMAXParams(spec=spec)
    p.exog_params = exog_params
    if spec.max_ar_order > 0:
        p.ar_params = np.zeros(spec.k_ar_params)
    if spec.max_seasonal_ar_order > 0:
        p.seasonal_ar_params = np.zeros(spec.k_seasonal_ar_params)
    if spec.max_ma_order > 0:
        p.ma_params = np.zeros(spec.k_ma_params)
    if spec.max_seasonal_ma_order > 0:
        p.seasonal_ma_params = np.zeros(spec.k_seasonal_ma_params)
    p.sigma2 = res_ols.scale
    ar_params = p.ar_params
    seasonal_ar_params = p.seasonal_ar_params
    ma_params = p.ma_params
    seasonal_ma_params = p.seasonal_ma_params
    sigma2 = p.sigma2
    arma_results = [None]
    differences = [None]
    parameters = [p]
    converged = False if n_iter is None else None
    i = 0

    def _check_arma_estimator_kwargs(kwargs, method):
        if False:
            for i in range(10):
                print('nop')
        if kwargs:
            raise ValueError(f'arma_estimator_kwargs not supported for method {method}')
    for i in range(1, max_iter + 1):
        prev = exog_params
        if arma_estimator == 'yule_walker':
            (p_arma, res_arma) = yule_walker(resid, ar_order=spec.ar_order, demean=False, **arma_estimator_kwargs)
        elif arma_estimator == 'burg':
            _check_arma_estimator_kwargs(arma_estimator_kwargs, 'burg')
            (p_arma, res_arma) = burg(resid, ar_order=spec.ar_order, demean=False)
        elif arma_estimator == 'innovations':
            _check_arma_estimator_kwargs(arma_estimator_kwargs, 'innovations')
            (out, res_arma) = innovations(resid, ma_order=spec.ma_order, demean=False)
            p_arma = out[-1]
        elif arma_estimator == 'hannan_rissanen':
            (p_arma, res_arma) = hannan_rissanen(resid, ar_order=spec.ar_order, ma_order=spec.ma_order, demean=False, **arma_estimator_kwargs)
        else:
            start_params = None if i == 1 else np.r_[ar_params, ma_params, seasonal_ar_params, seasonal_ma_params, sigma2]
            tmp_order = (spec.order[0], 0, spec.order[2])
            tmp_seasonal_order = (spec.seasonal_order[0], 0, spec.seasonal_order[2], spec.seasonal_order[3])
            if arma_estimator == 'innovations_mle':
                (p_arma, res_arma) = innovations_mle(resid, order=tmp_order, seasonal_order=tmp_seasonal_order, demean=False, start_params=start_params, **arma_estimator_kwargs)
            else:
                (p_arma, res_arma) = statespace(resid, order=tmp_order, seasonal_order=tmp_seasonal_order, include_constant=False, start_params=start_params, **arma_estimator_kwargs)
        ar_params = p_arma.ar_params
        seasonal_ar_params = p_arma.seasonal_ar_params
        ma_params = p_arma.ma_params
        seasonal_ma_params = p_arma.seasonal_ma_params
        sigma2 = p_arma.sigma2
        arma_results.append(res_arma)
        if not p_arma.is_stationary:
            raise ValueError('Roots of the autoregressive parameters indicate that data isnon-stationary. GLS cannot be used with non-stationary parameters. You should consider differencing the model dataor applying a nonlinear transformation (e.g., natural log).')
        (tmp, _) = arma_innovations.arma_innovations(augmented, ar_params=ar_params, ma_params=ma_params, normalize=True)
        u = tmp[:, 0]
        x = tmp[:, 1:]
        mod_gls = OLS(u, x)
        res_gls = mod_gls.fit()
        exog_params = res_gls.params
        resid = endog - np.dot(exog, exog_params)
        p = SARIMAXParams(spec=spec)
        p.exog_params = exog_params
        if spec.max_ar_order > 0:
            p.ar_params = ar_params
        if spec.max_seasonal_ar_order > 0:
            p.seasonal_ar_params = seasonal_ar_params
        if spec.max_ma_order > 0:
            p.ma_params = ma_params
        if spec.max_seasonal_ma_order > 0:
            p.seasonal_ma_params = seasonal_ma_params
        p.sigma2 = sigma2
        parameters.append(p)
        difference = np.abs(exog_params - prev)
        differences.append(difference)
        if n_iter is None and np.all(difference < tolerance):
            converged = True
            break
    else:
        if n_iter is None:
            warnings.warn('Feasible GLS failed to converge in %d iterations. Consider increasing the maximum number of iterations using the `max_iter` argument or reducing the required tolerance using the `tolerance` argument.' % max_iter)
    p = parameters[-1]
    other_results = Bunch({'spec': spec, 'params': parameters, 'converged': converged, 'differences': differences, 'iterations': i, 'arma_estimator': arma_estimator, 'arma_estimator_kwargs': arma_estimator_kwargs, 'arma_results': arma_results})
    return (p, other_results)