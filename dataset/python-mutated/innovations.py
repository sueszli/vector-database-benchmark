"""
Innovations algorithm for MA(q) and SARIMA(p,d,q)x(P,D,Q,s) model parameters.

Author: Chad Fulton
License: BSD-3
"""
import warnings
import numpy as np
from scipy.optimize import minimize
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.innovations import arma_innovations
from statsmodels.tsa.stattools import acovf, innovations_algo
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen

def innovations(endog, ma_order=0, demean=True):
    if False:
        while True:
            i = 10
    '\n    Estimate MA parameters using innovations algorithm.\n\n    Parameters\n    ----------\n    endog : array_like or SARIMAXSpecification\n        Input time series array, assumed to be stationary.\n    ma_order : int, optional\n        Maximum moving average order. Default is 0.\n    demean : bool, optional\n        Whether to estimate and remove the mean from the process prior to\n        fitting the moving average coefficients. Default is True.\n\n    Returns\n    -------\n    parameters : list of SARIMAXParams objects\n        List elements correspond to estimates at different `ma_order`. For\n        example, parameters[0] is an `SARIMAXParams` instance corresponding to\n        `ma_order=0`.\n    other_results : Bunch\n        Includes one component, `spec`, containing the `SARIMAXSpecification`\n        instance corresponding to the input arguments.\n\n    Notes\n    -----\n    The primary reference is [1]_, section 5.1.3.\n\n    This procedure assumes that the series is stationary.\n\n    References\n    ----------\n    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.\n       Introduction to Time Series and Forecasting. Springer.\n    '
    spec = max_spec = SARIMAXSpecification(endog, ma_order=ma_order)
    endog = max_spec.endog
    if demean:
        endog = endog - endog.mean()
    if not max_spec.is_ma_consecutive:
        raise ValueError('Innovations estimation unavailable for models with seasonal or otherwise non-consecutive MA orders.')
    sample_acovf = acovf(endog, fft=True)
    (theta, v) = innovations_algo(sample_acovf, nobs=max_spec.ma_order + 1)
    ma_params = [theta[i, :i] for i in range(1, max_spec.ma_order + 1)]
    sigma2 = v
    out = []
    for i in range(max_spec.ma_order + 1):
        spec = SARIMAXSpecification(ma_order=i)
        p = SARIMAXParams(spec=spec)
        if i == 0:
            p.params = sigma2[i]
        else:
            p.params = np.r_[ma_params[i - 1], sigma2[i]]
        out.append(p)
    other_results = Bunch({'spec': spec})
    return (out, other_results)

def innovations_mle(endog, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0), demean=True, enforce_invertibility=True, start_params=None, minimize_kwargs=None):
    if False:
        return 10
    "\n    Estimate SARIMA parameters by MLE using innovations algorithm.\n\n    Parameters\n    ----------\n    endog : array_like\n        Input time series array.\n    order : tuple, optional\n        The (p,d,q) order of the model for the number of AR parameters,\n        differences, and MA parameters. Default is (0, 0, 0).\n    seasonal_order : tuple, optional\n        The (P,D,Q,s) order of the seasonal component of the model for the\n        AR parameters, differences, MA parameters, and periodicity. Default\n        is (0, 0, 0, 0).\n    demean : bool, optional\n        Whether to estimate and remove the mean from the process prior to\n        fitting the SARIMA coefficients. Default is True.\n    enforce_invertibility : bool, optional\n        Whether or not to transform the MA parameters to enforce invertibility\n        in the moving average component of the model. Default is True.\n    start_params : array_like, optional\n        Initial guess of the solution for the loglikelihood maximization. The\n        AR polynomial must be stationary. If `enforce_invertibility=True` the\n        MA poylnomial must be invertible. If not provided, default starting\n        parameters are computed using the Hannan-Rissanen method.\n    minimize_kwargs : dict, optional\n        Arguments to pass to scipy.optimize.minimize.\n\n    Returns\n    -------\n    parameters : SARIMAXParams object\n    other_results : Bunch\n        Includes four components: `spec`, containing the `SARIMAXSpecification`\n        instance corresponding to the input arguments; `minimize_kwargs`,\n        containing any keyword arguments passed to `minimize`; `start_params`,\n        containing the untransformed starting parameters passed to `minimize`;\n        and `minimize_results`, containing the output from `minimize`.\n\n    Notes\n    -----\n    The primary reference is [1]_, section 5.2.\n\n    Note: we do not include `enforce_stationarity` as an argument, because this\n    function requires stationarity.\n\n    TODO: support concentrating out the scale (should be easy: use sigma2=1\n          and then compute sigma2=np.sum(u**2 / v) / len(u); would then need to\n          redo llf computation in the Cython function).\n\n    TODO: add support for fixed parameters\n\n    TODO: add support for secondary optimization that does not enforce\n          stationarity / invertibility, starting from first step's parameters\n\n    References\n    ----------\n    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.\n       Introduction to Time Series and Forecasting. Springer.\n    "
    spec = SARIMAXSpecification(endog, order=order, seasonal_order=seasonal_order, enforce_stationarity=True, enforce_invertibility=enforce_invertibility)
    endog = spec.endog
    if spec.is_integrated:
        warnings.warn('Provided `endog` series has been differenced to eliminate integration prior to ARMA parameter estimation.')
        endog = diff(endog, k_diff=spec.diff, k_seasonal_diff=spec.seasonal_diff, seasonal_periods=spec.seasonal_periods)
    if demean:
        endog = endog - endog.mean()
    p = SARIMAXParams(spec=spec)
    if start_params is None:
        sp = SARIMAXParams(spec=spec)
        (hr, hr_results) = hannan_rissanen(endog, ar_order=spec.ar_order, ma_order=spec.ma_order, demean=False)
        if spec.seasonal_periods == 0:
            sp.params = hr.params
        else:
            _ = SARIMAXSpecification(endog, seasonal_order=seasonal_order, enforce_stationarity=True, enforce_invertibility=enforce_invertibility)
            ar_order = np.array(spec.seasonal_ar_lags) * spec.seasonal_periods
            ma_order = np.array(spec.seasonal_ma_lags) * spec.seasonal_periods
            (seasonal_hr, seasonal_hr_results) = hannan_rissanen(hr_results.resid, ar_order=ar_order, ma_order=ma_order, demean=False)
            sp.ar_params = hr.ar_params
            sp.ma_params = hr.ma_params
            sp.seasonal_ar_params = seasonal_hr.ar_params
            sp.seasonal_ma_params = seasonal_hr.ma_params
            sp.sigma2 = seasonal_hr.sigma2
        if not sp.is_stationary:
            sp.ar_params = [0] * sp.k_ar_params
            sp.seasonal_ar_params = [0] * sp.k_seasonal_ar_params
        if not sp.is_invertible and spec.enforce_invertibility:
            sp.ma_params = [0] * sp.k_ma_params
            sp.seasonal_ma_params = [0] * sp.k_seasonal_ma_params
        start_params = sp.params
    else:
        sp = SARIMAXParams(spec=spec)
        sp.params = start_params
        if not sp.is_stationary:
            raise ValueError('Given starting parameters imply a non-stationary AR process. Innovations algorithm requires a stationary process.')
        if spec.enforce_invertibility and (not sp.is_invertible):
            raise ValueError('Given starting parameters imply a non-invertible MA process with `enforce_invertibility=True`.')

    def obj(params):
        if False:
            while True:
                i = 10
        p.params = spec.constrain_params(params)
        return -arma_innovations.arma_loglike(endog, ar_params=-p.reduced_ar_poly.coef[1:], ma_params=p.reduced_ma_poly.coef[1:], sigma2=p.sigma2)
    unconstrained_start_params = spec.unconstrain_params(start_params)
    if minimize_kwargs is None:
        minimize_kwargs = {}
    if 'options' not in minimize_kwargs:
        minimize_kwargs['options'] = {}
    minimize_kwargs['options'].setdefault('maxiter', 100)
    minimize_results = minimize(obj, unconstrained_start_params, **minimize_kwargs)
    p.params = spec.constrain_params(minimize_results.x)
    other_results = Bunch({'spec': spec, 'minimize_results': minimize_results, 'minimize_kwargs': minimize_kwargs, 'start_params': start_params})
    return (p, other_results)