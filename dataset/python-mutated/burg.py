"""
Burg's method for estimating AR(p) model parameters.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
from statsmodels.tools.tools import Bunch
from statsmodels.regression import linear_model
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams

def burg(endog, ar_order=0, demean=True):
    if False:
        print('Hello World!')
    '\n    Estimate AR parameters using Burg technique.\n\n    Parameters\n    ----------\n    endog : array_like or SARIMAXSpecification\n        Input time series array, assumed to be stationary.\n    ar_order : int, optional\n        Autoregressive order. Default is 0.\n    demean : bool, optional\n        Whether to estimate and remove the mean from the process prior to\n        fitting the autoregressive coefficients.\n\n    Returns\n    -------\n    parameters : SARIMAXParams object\n        Contains the parameter estimates from the final iteration.\n    other_results : Bunch\n        Includes one component, `spec`, which is the `SARIMAXSpecification`\n        instance corresponding to the input arguments.\n\n    Notes\n    -----\n    The primary reference is [1]_, section 5.1.2.\n\n    This procedure assumes that the series is stationary.\n\n    This function is a light wrapper around `statsmodels.linear_model.burg`.\n\n    References\n    ----------\n    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.\n       Introduction to Time Series and Forecasting. Springer.\n    '
    spec = SARIMAXSpecification(endog, ar_order=ar_order)
    endog = spec.endog
    if np.issubdtype(endog.dtype, np.dtype(int)):
        endog = endog * 1.0
    if not spec.is_ar_consecutive:
        raise ValueError('Burg estimation unavailable for models with seasonal or otherwise non-consecutive AR orders.')
    p = SARIMAXParams(spec=spec)
    if ar_order == 0:
        p.sigma2 = np.var(endog)
    else:
        (p.ar_params, p.sigma2) = linear_model.burg(endog, order=ar_order, demean=demean)
    other_results = Bunch({'spec': spec})
    return (p, other_results)