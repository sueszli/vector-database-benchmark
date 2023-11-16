import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.tsa.stattools import acovf
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker

@pytest.mark.low_precision('Test against Example 5.1.1 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_511():
    if False:
        for i in range(10):
            print('nop')
    endog = dowj.diff().iloc[1:]
    assert_equal(len(endog), 77)
    desired = [0.17992, 0.0759, 0.04885]
    assert_allclose(acovf(endog, fft=True, nlag=2), desired, atol=1e-05)
    (yw, _) = yule_walker(endog, ar_order=1, demean=True)
    assert_allclose(yw.ar_params, [0.4219], atol=0.0001)
    assert_allclose(yw.sigma2, 0.1479, atol=0.0001)

@pytest.mark.low_precision('Test against Example 5.1.4 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_514():
    if False:
        return 10
    endog = lake.copy()
    (res, _) = yule_walker(endog, ar_order=2, demean=True)
    assert_allclose(res.ar_params, [1.0538, -0.2668], atol=0.0001)
    assert_allclose(res.sigma2, 0.492, atol=0.0001)

def check_itsmr(lake):
    if False:
        i = 10
        return i + 15
    (yw, _) = yule_walker(lake, 5)
    desired = [1.08213598501, -0.39658257147, 0.11793957728, -0.03326633983, 0.06209208707]
    assert_allclose(yw.ar_params, desired)
    (u, v) = arma_innovations(np.array(lake) - np.mean(lake), ar_params=yw.ar_params, sigma2=1)
    desired_sigma2 = 0.4716322564
    assert_allclose(np.sum(u ** 2 / v) / len(u), desired_sigma2)

def test_itsmr():
    if False:
        print('Hello World!')
    endog = lake.copy()
    check_itsmr(endog)
    check_itsmr(endog.values)
    check_itsmr(endog.tolist())

def test_invalid():
    if False:
        print('Hello World!')
    endog = np.arange(2) * 1.0
    assert_raises(ValueError, yule_walker, endog, ar_order=-1)
    assert_raises(ValueError, yule_walker, endog, ar_order=1.5)
    endog = np.arange(10) * 1.0
    assert_raises(ValueError, yule_walker, endog, ar_order=[1, 3])

@pytest.mark.xfail(reason='TODO: this does not raise an error due to the way linear_model.yule_walker works.')
def test_invalid_xfail():
    if False:
        for i in range(10):
            print('nop')
    endog = np.arange(2) * 1.0
    assert_raises(ValueError, yule_walker, endog, ar_order=2)