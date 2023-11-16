"""
Yule-Walker method for estimating AR(p) model parameters.

Author: Chad Fulton
License: BSD-3
"""
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.regression import linear_model
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.arima.specification import SARIMAXSpecification

@deprecate_kwarg('unbiased', 'adjusted')
def yule_walker(endog, ar_order=0, demean=True, adjusted=False):
    if False:
        return 10
    '\n    Estimate AR parameters using Yule-Walker equations.\n\n    Parameters\n    ----------\n    endog : array_like or SARIMAXSpecification\n        Input time series array, assumed to be stationary.\n    ar_order : int, optional\n        Autoregressive order. Default is 0.\n    demean : bool, optional\n        Whether to estimate and remove the mean from the process prior to\n        fitting the autoregressive coefficients. Default is True.\n    adjusted : bool, optional\n        Whether to use the adjusted autocovariance estimator, which uses\n        n - h degrees of freedom rather than n. For some processes this option\n        may  result in a non-positive definite autocovariance matrix. Default\n        is False.\n\n    Returns\n    -------\n    parameters : SARIMAXParams object\n        Contains the parameter estimates from the final iteration.\n    other_results : Bunch\n        Includes one component, `spec`, which is the `SARIMAXSpecification`\n        instance corresponding to the input arguments.\n\n    Notes\n    -----\n    The primary reference is [1]_, section 5.1.1.\n\n    This procedure assumes that the series is stationary.\n\n    For a description of the effect of the adjusted estimate of the\n    autocovariance function, see 2.4.2 of [1]_.\n\n    References\n    ----------\n    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.\n       Introduction to Time Series and Forecasting. Springer.\n    '
    spec = SARIMAXSpecification(endog, ar_order=ar_order)
    endog = spec.endog
    p = SARIMAXParams(spec=spec)
    if not spec.is_ar_consecutive:
        raise ValueError('Yule-Walker estimation unavailable for models with seasonal or non-consecutive AR orders.')
    method = 'adjusted' if adjusted else 'mle'
    (p.ar_params, sigma) = linear_model.yule_walker(endog, order=ar_order, demean=demean, method=method)
    p.sigma2 = sigma ** 2
    other_results = Bunch({'spec': spec})
    return (p, other_results)