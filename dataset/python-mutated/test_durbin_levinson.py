import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_raises
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.durbin_levinson import durbin_levinson

@pytest.mark.low_precision('Test against Example 5.1.1 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_511():
    if False:
        while True:
            i = 10
    endog = dowj.diff().iloc[1:]
    (dl, _) = durbin_levinson(endog, ar_order=2, demean=True)
    assert_allclose(dl[0].params, np.var(endog))
    assert_allclose(dl[1].params, [0.4219, 0.1479], atol=0.0001)
    assert_allclose(dl[2].params, [0.3739, 0.1138, 0.146], atol=0.0001)

def check_itsmr(lake):
    if False:
        for i in range(10):
            print('nop')
    (dl, _) = durbin_levinson(lake, 5)
    assert_allclose(dl[0].params, np.var(lake))
    assert_allclose(dl[1].ar_params, [0.8319112104])
    assert_allclose(dl[2].ar_params, [1.0538248798, -0.2667516276])
    desired = [1.0887037577, -0.4045435867, 0.1307541335]
    assert_allclose(dl[3].ar_params, desired)
    desired = [1.0842506581, -0.39076602696, 0.09367609911, 0.03405704644]
    assert_allclose(dl[4].ar_params, desired)
    desired = [1.08213598501, -0.39658257147, 0.11793957728, -0.03326633983, 0.06209208707]
    assert_allclose(dl[5].ar_params, desired)
    (u, v) = arma_innovations(np.array(lake) - np.mean(lake), ar_params=dl[5].ar_params, sigma2=1)
    desired_sigma2 = 0.4716322564
    assert_allclose(np.sum(u ** 2 / v) / len(u), desired_sigma2)

def test_itsmr():
    if False:
        for i in range(10):
            print('nop')
    endog = lake.copy()
    check_itsmr(endog)
    check_itsmr(endog.values)
    check_itsmr(endog.tolist())

def test_nonstationary_series():
    if False:
        i = 10
        return i + 15
    endog = np.arange(1, 12) * 1.0
    (res, _) = durbin_levinson(endog, 2, demean=False)
    desired_ar_params = [0.92318534179, -0.06166314306]
    assert_allclose(res[2].ar_params, desired_ar_params)

@pytest.mark.xfail(reason='Different computation of variances')
def test_nonstationary_series_variance():
    if False:
        return 10
    endog = np.arange(1, 12) * 1.0
    (res, _) = durbin_levinson(endog, 2, demean=False)
    desired_sigma2 = 15.36526603
    assert_allclose(res[2].sigma2, desired_sigma2)

def test_invalid():
    if False:
        while True:
            i = 10
    endog = np.arange(2) * 1.0
    assert_raises(ValueError, durbin_levinson, endog, ar_order=2)
    assert_raises(ValueError, durbin_levinson, endog, ar_order=-1)
    assert_raises(ValueError, durbin_levinson, endog, ar_order=1.5)
    endog = np.arange(10) * 1.0
    assert_raises(ValueError, durbin_levinson, endog, ar_order=[1, 3])

def test_misc():
    if False:
        print('Hello World!')
    endog = lake.copy()
    (res, _) = durbin_levinson(endog)
    assert_allclose(res[0].params, np.var(endog))
    endog = np.array([1, 2, 5, 3, -2, 1, -3, 5, 2, 3, -1], dtype=int)
    (res_int, _) = durbin_levinson(endog, 2, demean=False)
    (res_float, _) = durbin_levinson(endog * 1.0, 2, demean=False)
    assert_allclose(res_int[0].params, res_float[0].params)
    assert_allclose(res_int[1].params, res_float[1].params)
    assert_allclose(res_int[2].params, res_float[2].params)