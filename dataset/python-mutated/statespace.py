"""
State space approach to estimating SARIMAX models.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
from statsmodels.tools.tools import add_constant, Bunch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams

def statespace(endog, exog=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0), include_constant=True, enforce_stationarity=True, enforce_invertibility=True, concentrate_scale=False, start_params=None, fit_kwargs=None):
    if False:
        while True:
            i = 10
    "\n    Estimate SARIMAX parameters using state space methods.\n\n    Parameters\n    ----------\n    endog : array_like\n        Input time series array.\n    order : tuple, optional\n        The (p,d,q) order of the model for the number of AR parameters,\n        differences, and MA parameters. Default is (0, 0, 0).\n    seasonal_order : tuple, optional\n        The (P,D,Q,s) order of the seasonal component of the model for the\n        AR parameters, differences, MA parameters, and periodicity. Default\n        is (0, 0, 0, 0).\n    include_constant : bool, optional\n        Whether to add a constant term in `exog` if it's not already there.\n        The estimate of the constant will then appear as one of the `exog`\n        parameters. If `exog` is None, then the constant will represent the\n        mean of the process.\n    enforce_stationarity : bool, optional\n        Whether or not to transform the AR parameters to enforce stationarity\n        in the autoregressive component of the model. Default is True.\n    enforce_invertibility : bool, optional\n        Whether or not to transform the MA parameters to enforce invertibility\n        in the moving average component of the model. Default is True.\n    concentrate_scale : bool, optional\n        Whether or not to concentrate the scale (variance of the error term)\n        out of the likelihood. This reduces the number of parameters estimated\n        by maximum likelihood by one.\n    start_params : array_like, optional\n        Initial guess of the solution for the loglikelihood maximization. The\n        AR polynomial must be stationary. If `enforce_invertibility=True` the\n        MA poylnomial must be invertible. If not provided, default starting\n        parameters are computed using the Hannan-Rissanen method.\n    fit_kwargs : dict, optional\n        Arguments to pass to the state space model's `fit` method.\n\n    Returns\n    -------\n    parameters : SARIMAXParams object\n    other_results : Bunch\n        Includes two components, `spec`, containing the `SARIMAXSpecification`\n        instance corresponding to the input arguments; and\n        `state_space_results`, corresponding to the results from the underlying\n        state space model and Kalman filter / smoother.\n\n    Notes\n    -----\n    The primary reference is [1]_.\n\n    References\n    ----------\n    .. [1] Durbin, James, and Siem Jan Koopman. 2012.\n       Time Series Analysis by State Space Methods: Second Edition.\n       Oxford University Press.\n    "
    if include_constant:
        exog = np.ones_like(endog) if exog is None else add_constant(exog)
    spec = SARIMAXSpecification(endog, exog=exog, order=order, seasonal_order=seasonal_order, enforce_stationarity=enforce_stationarity, enforce_invertibility=enforce_invertibility, concentrate_scale=concentrate_scale)
    endog = spec.endog
    exog = spec.exog
    p = SARIMAXParams(spec=spec)
    if start_params is not None:
        sp = SARIMAXParams(spec=spec)
        sp.params = start_params
        if spec.enforce_stationarity and (not sp.is_stationary):
            raise ValueError('Given starting parameters imply a non-stationary AR process with `enforce_stationarity=True`.')
        if spec.enforce_invertibility and (not sp.is_invertible):
            raise ValueError('Given starting parameters imply a non-invertible MA process with `enforce_invertibility=True`.')
    mod = SARIMAX(endog, exog=exog, order=spec.order, seasonal_order=spec.seasonal_order, enforce_stationarity=spec.enforce_stationarity, enforce_invertibility=spec.enforce_invertibility, concentrate_scale=spec.concentrate_scale)
    if fit_kwargs is None:
        fit_kwargs = {}
    fit_kwargs.setdefault('disp', 0)
    res_ss = mod.fit(start_params=start_params, **fit_kwargs)
    p.params = res_ss.params
    res = Bunch({'spec': spec, 'statespace_results': res_ss})
    return (p, res)