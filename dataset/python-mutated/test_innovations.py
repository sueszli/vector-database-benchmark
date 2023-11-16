import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_warns, assert_raises
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake, oshorts
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import innovations, innovations_mle

@pytest.mark.low_precision('Test against Example 5.1.5 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_515():
    if False:
        print('Hello World!')
    endog = dowj.diff().iloc[1:]
    (p, _) = innovations(endog, ma_order=17, demean=True)
    assert_allclose(p[17].ma_params[:2], [0.4269, 0.2704], atol=0.0001)
    assert_allclose(p[17].sigma2, 0.1122, atol=0.0001)
    desired = [0.4269, 0.2704, 0.1183, 0.1589, 0.1355, 0.1568, 0.1284, -0.006, 0.0148, -0.0017, 0.1974, -0.0463, 0.2023, 0.1285, -0.0213, -0.2575, 0.076]
    assert_allclose(p[17].ma_params, desired, atol=0.0001)

def check_innovations_ma_itsmr(lake):
    if False:
        return 10
    (ia, _) = innovations(lake, 10, demean=True)
    desired = [1.0816255264, 0.7781248438, 0.536716443, 0.3291559246, 0.316003985, 0.251375455, 0.2051536531, 0.1441070313, 0.343186834, 0.1827400798]
    assert_allclose(ia[10].ma_params, desired)
    (u, v) = arma_innovations(np.array(lake) - np.mean(lake), ma_params=ia[10].ma_params, sigma2=1)
    desired_sigma2 = 0.4523684344
    assert_allclose(np.sum(u ** 2 / v) / len(u), desired_sigma2)

def test_innovations_ma_itsmr():
    if False:
        return 10
    endog = lake.copy()
    check_innovations_ma_itsmr(endog)
    check_innovations_ma_itsmr(endog.values)
    check_innovations_ma_itsmr(endog.tolist())

def test_innovations_ma_invalid():
    if False:
        for i in range(10):
            print('nop')
    endog = np.arange(2)
    assert_raises(ValueError, innovations, endog, ma_order=2)
    assert_raises(ValueError, innovations, endog, ma_order=-1)
    assert_raises(ValueError, innovations, endog, ma_order=1.5)
    endog = np.arange(10)
    assert_raises(ValueError, innovations, endog, ma_order=[1, 3])

@pytest.mark.low_precision('Test against Example 5.2.4 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_524():
    if False:
        return 10
    endog = dowj.diff().iloc[1:]
    (initial, _) = burg(endog, ar_order=1, demean=True)
    (p, _) = innovations_mle(endog, order=(1, 0, 0), demean=True, start_params=initial.params)
    assert_allclose(p.ar_params, 0.4471, atol=0.0001)

@pytest.mark.low_precision('Test against Example 5.2.4 in Brockwell and Davis (2016)')
@pytest.mark.xfail(reason='Suspicious result reported in Brockwell and Davis (2016).')
def test_brockwell_davis_example_524_variance():
    if False:
        while True:
            i = 10
    endog = dowj.diff().iloc[1:]
    (initial, _) = burg(endog, ar_order=1, demean=True)
    (p, _) = innovations_mle(endog, order=(1, 0, 0), demean=True, start_params=initial.params)
    assert_allclose(p.sigma2, 0.02117, atol=0.0001)

@pytest.mark.low_precision('Test against Example 5.2.5 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_525():
    if False:
        for i in range(10):
            print('nop')
    endog = lake.copy()
    (initial, _) = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True)
    (p, _) = innovations_mle(endog, order=(1, 0, 1), demean=True, start_params=initial.params)
    assert_allclose(p.params, [0.7446, 0.3213, 0.475], atol=0.0001)
    (p, _) = innovations_mle(endog, order=(1, 0, 1), demean=True)
    assert_allclose(p.params, [0.7446, 0.3213, 0.475], atol=0.0001)

@pytest.mark.low_precision('Test against Example 5.4.1 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_541():
    if False:
        i = 10
        return i + 15
    endog = oshorts.copy()
    (initial, _) = innovations(endog, ma_order=1, demean=True)
    (p, _) = innovations_mle(endog, order=(0, 0, 1), demean=True, start_params=initial[1].params)
    assert_allclose(p.ma_params, -0.818, atol=0.001)

def test_innovations_mle_statespace():
    if False:
        i = 10
        return i + 15
    endog = lake.copy()
    endog = endog - endog.mean()
    start_params = [0, 0, np.var(endog)]
    (p, mleres) = innovations_mle(endog, order=(1, 0, 1), demean=False, start_params=start_params)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 1))
    res = mod.filter(p.params)
    assert_allclose(-mleres.minimize_results.fun, res.llf)
    res2 = mod.fit(start_params=p.params, disp=0)
    assert_allclose(p.params, res2.params)
    (p2, _) = innovations_mle(endog, order=(1, 0, 1), demean=False)
    assert_allclose(p.params, p2.params, atol=1e-05)

def test_innovations_mle_statespace_seasonal():
    if False:
        for i in range(10):
            print('nop')
    endog = lake.copy()
    endog = endog - endog.mean()
    start_params = [0, np.var(endog)]
    (p, mleres) = innovations_mle(endog, seasonal_order=(1, 0, 0, 4), demean=False, start_params=start_params)
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), seasonal_order=(1, 0, 0, 4))
    res = mod.filter(p.params)
    assert_allclose(-mleres.minimize_results.fun, res.llf)
    res2 = mod.fit(start_params=p.params, disp=0)
    assert_allclose(p.params, res2.params)
    (p2, _) = innovations_mle(endog, seasonal_order=(1, 0, 0, 4), demean=False)
    assert_allclose(p.params, p2.params, atol=1e-05)

def test_innovations_mle_statespace_nonconsecutive():
    if False:
        print('Hello World!')
    endog = lake.copy()
    endog = endog - endog.mean()
    start_params = [0, 0, np.var(endog)]
    (p, mleres) = innovations_mle(endog, order=([0, 1], 0, [0, 1]), demean=False, start_params=start_params)
    mod = sarimax.SARIMAX(endog, order=([0, 1], 0, [0, 1]))
    res = mod.filter(p.params)
    assert_allclose(-mleres.minimize_results.fun, res.llf)
    res2 = mod.fit(start_params=p.params, disp=0)
    assert_allclose(p.params, res2.params)
    (p2, _) = innovations_mle(endog, order=([0, 1], 0, [0, 1]), demean=False)
    assert_allclose(p.params, p2.params, atol=1e-05)

def test_innovations_mle_integrated():
    if False:
        for i in range(10):
            print('nop')
    endog = np.r_[0, np.cumsum(lake.copy())]
    start_params = [0, np.var(lake.copy())]
    with assert_warns(UserWarning):
        (p, mleres) = innovations_mle(endog, order=(1, 1, 0), demean=False, start_params=start_params)
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)
    res = mod.filter(p.params)
    assert_allclose(-mleres.minimize_results.fun, res.llf)
    res2 = mod.fit(start_params=p.params, disp=0)
    assert_allclose(p.params, res2.params, atol=1e-06)
    (p2, _) = innovations_mle(lake.copy(), order=(1, 0, 0), demean=False, start_params=start_params)
    assert_allclose(p.params, p2.params, atol=1e-05)

def test_innovations_mle_misc():
    if False:
        for i in range(10):
            print('nop')
    endog = np.arange(20) ** 2 * 1.0
    (hr, _) = hannan_rissanen(endog, ar_order=1, demean=False)
    assert_(hr.ar_params[0] > 1)
    (_, res) = innovations_mle(endog, order=(1, 0, 0))
    assert_allclose(res.start_params[0], 0)
    (hr, _) = hannan_rissanen(endog, ma_order=1, demean=False)
    assert_(hr.ma_params[0] > 1)
    (_, res) = innovations_mle(endog, order=(0, 0, 1))
    assert_allclose(res.start_params[0], 0)

def test_innovations_mle_invalid():
    if False:
        print('Hello World!')
    endog = np.arange(2) * 1.0
    assert_raises(ValueError, innovations_mle, endog, order=(0, 0, 2))
    assert_raises(ValueError, innovations_mle, endog, order=(0, 0, -1))
    assert_raises(ValueError, innovations_mle, endog, order=(0, 0, 1.5))
    endog = lake.copy()
    assert_raises(ValueError, innovations_mle, endog, order=(1, 0, 0), start_params=[1.0, 1.0])
    assert_raises(ValueError, innovations_mle, endog, order=(0, 0, 1), start_params=[1.0, 1.0])