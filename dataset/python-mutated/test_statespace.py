import numpy as np
from numpy.testing import assert_allclose, assert_raises
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.statespace import statespace

def test_basic():
    if False:
        i = 10
        return i + 15
    endog = lake.copy()
    exog = np.arange(1, len(endog) + 1) * 1.0
    (p, res) = statespace(endog, exog=exog, order=(1, 0, 0), include_constant=True, concentrate_scale=False)
    mod_ss = sarimax.SARIMAX(endog, exog=add_constant(exog), order=(1, 0, 0))
    res_ss = mod_ss.filter(p.params)
    assert_allclose(res.statespace_results.llf, res_ss.llf)
    (p, res) = statespace(endog, exog=exog, order=(1, 0, 0), include_constant=False, concentrate_scale=False)
    mod_ss = sarimax.SARIMAX(endog, exog=exog, order=(1, 0, 0))
    res_ss = mod_ss.filter(p.params)
    assert_allclose(res.statespace_results.llf, res_ss.llf)
    (p, res) = statespace(endog, exog=exog, order=(1, 0, 0), include_constant=True, concentrate_scale=True)
    mod_ss = sarimax.SARIMAX(endog, exog=add_constant(exog), order=(1, 0, 0), concentrate_scale=True)
    res_ss = mod_ss.filter(p.params)
    assert_allclose(res.statespace_results.llf, res_ss.llf)

def test_start_params():
    if False:
        print('Hello World!')
    endog = lake.copy()
    (p, _) = statespace(endog, order=(1, 0, 0), start_params=[0, 0, 1.0])
    (p, _) = statespace(endog, order=(1, 0, 0), start_params=[0, 1.0, 1.0], enforce_stationarity=False)
    (p, _) = statespace(endog, order=(0, 0, 1), start_params=[0, 1.0, 1.0], enforce_invertibility=False)
    assert_raises(ValueError, statespace, endog, order=(1, 0, 0), start_params=[0, 1.0, 1.0])
    assert_raises(ValueError, statespace, endog, order=(0, 0, 1), start_params=[0, 1.0, 1.0])