import numpy as np
import pandas as pd
import pytest
from pypfopt import expected_returns
from tests.utilities_for_tests import get_benchmark_data, get_data

def test_returns_dataframe():
    if False:
        for i in range(10):
            print('nop')
    df = get_data()
    returns_df = expected_returns.returns_from_prices(df)
    assert isinstance(returns_df, pd.DataFrame)
    assert returns_df.shape[1] == 20
    assert len(returns_df) == 7125
    assert not ((returns_df > 1) & returns_df.notnull()).any().any()

def test_prices_from_returns():
    if False:
        return 10
    df = get_data()
    returns_df = df.pct_change()
    pseudo_prices = expected_returns.prices_from_returns(returns_df)
    initial_prices = df.bfill().iloc[0]
    test_prices = pseudo_prices * initial_prices
    assert ((test_prices[1:] - df[1:]).fillna(0) < 1e-10).all().all()

def test_prices_from_log_returns():
    if False:
        i = 10
        return i + 15
    df = get_data()
    returns_df = df.pct_change()
    log_returns_df = np.log1p(returns_df)
    pseudo_prices = expected_returns.prices_from_returns(log_returns_df, log_returns=True)
    initial_prices = df.bfill().iloc[0]
    test_prices = pseudo_prices * initial_prices
    assert ((test_prices[1:] - df[1:]).fillna(0) < 1e-05).all().all()

def test_returns_from_prices():
    if False:
        return 10
    df = get_data()
    returns_df = expected_returns.returns_from_prices(df)
    pd.testing.assert_series_equal(returns_df.iloc[-1], df.pct_change().iloc[-1])

def test_returns_warning():
    if False:
        i = 10
        return i + 15
    df = get_data()
    df.iloc[3, :] = 0
    with pytest.warns(UserWarning):
        expected_returns.mean_historical_return(df)

def test_log_returns_from_prices():
    if False:
        return 10
    df = get_data()
    old_nan = df.isnull().sum(axis=1).sum()
    log_rets = expected_returns.returns_from_prices(df, log_returns=True)
    new_nan = log_rets.isnull().sum(axis=1).sum()
    assert new_nan == old_nan
    np.testing.assert_almost_equal(log_rets.iloc[-1, -1], 0.0001682740081102576)

def test_mean_historical_returns_dummy():
    if False:
        return 10
    data = pd.DataFrame([[4.0, 2.0, 0.6, -12], [4.2, 2.1, 0.59, -13.2], [3.9, 2.0, 0.58, -11.3], [4.3, 2.1, 0.62, -11.7], [4.1, 2.2, 0.63, -10.1]])
    mean = expected_returns.mean_historical_return(data, frequency=1)
    test_answer = pd.Series([0.0061922, 0.0241137, 0.0122722, -0.0421775])
    pd.testing.assert_series_equal(mean, test_answer, rtol=0.001)
    mean = expected_returns.mean_historical_return(data, compounding=False, frequency=1)
    test_answer = pd.Series([0.008656, 0.025, 0.0128697, -0.03632333])
    pd.testing.assert_series_equal(mean, test_answer, rtol=0.001)

def test_mean_historical_returns():
    if False:
        for i in range(10):
            print('nop')
    df = get_data()
    mean = expected_returns.mean_historical_return(df)
    assert isinstance(mean, pd.Series)
    assert list(mean.index) == list(df.columns)
    assert mean.notnull().all()
    assert mean.dtype == 'float64'
    correct_mean = np.array([0.247967, 0.294304, 0.284037, 0.1923164, 0.371327, 0.1360093, 0.0328503, 0.1200115, 0.10554, 0.0423457, 0.1002559, 0.1442237, -0.0792602, 0.1430506, 0.0736356, 0.238835, 0.388665, 0.226717, 0.1561701, 0.2318153])
    np.testing.assert_array_almost_equal(mean.values, correct_mean)

def test_mean_historical_returns_type_warning():
    if False:
        while True:
            i = 10
    df = get_data()
    mean = expected_returns.mean_historical_return(df)
    with pytest.warns(RuntimeWarning) as w:
        mean_from_array = expected_returns.mean_historical_return(np.array(df))
        assert len(w) == 1
        assert str(w[0].message) == 'prices are not in a dataframe'
    np.testing.assert_array_almost_equal(mean.values, mean_from_array.values, decimal=6)

def test_mean_historical_returns_frequency():
    if False:
        while True:
            i = 10
    df = get_data()
    mean = expected_returns.mean_historical_return(df, compounding=False)
    mean2 = expected_returns.mean_historical_return(df, compounding=False, frequency=52)
    np.testing.assert_array_almost_equal(mean / 252, mean2 / 52)

def test_ema_historical_return():
    if False:
        i = 10
        return i + 15
    df = get_data()
    mean = expected_returns.ema_historical_return(df)
    assert isinstance(mean, pd.Series)
    assert list(mean.index) == list(df.columns)
    assert mean.notnull().all()
    assert mean.dtype == 'float64'
    with pytest.warns(RuntimeWarning):
        mean_np = expected_returns.ema_historical_return(df.to_numpy())
        mean_np.name = mean.name
        reset_mean = mean.reset_index(drop=True)
        pd.testing.assert_series_equal(mean_np, reset_mean)

def test_ema_historical_return_frequency():
    if False:
        i = 10
        return i + 15
    df = get_data()
    mean = expected_returns.ema_historical_return(df, compounding=False)
    mean2 = expected_returns.ema_historical_return(df, compounding=False, frequency=52)
    np.testing.assert_array_almost_equal(mean / 252, mean2 / 52)

def test_ema_historical_return_limit():
    if False:
        while True:
            i = 10
    df = get_data()
    sma = expected_returns.mean_historical_return(df, compounding=False)
    ema = expected_returns.ema_historical_return(df, compounding=False, span=10000000000.0)
    np.testing.assert_array_almost_equal(ema.values, sma.values)

def test_capm_no_benchmark():
    if False:
        for i in range(10):
            print('nop')
    df = get_data()
    mu = expected_returns.capm_return(df)
    assert isinstance(mu, pd.Series)
    assert list(mu.index) == list(df.columns)
    assert mu.notnull().all()
    assert mu.dtype == 'float64'
    correct_mu = np.array([0.22148462799238577, 0.2835429647498704, 0.14693081977908462, 0.1488989354304723, 0.4162399750335195, 0.22716772604184535, 0.3970337136813829, 0.16733214988182069, 0.31791477659742146, 0.17279931642386534, 0.15271750464365566, 0.351778014382922, 0.32859883451716376, 0.1501938182844417, 0.268295486802897, 0.31632339201710874, 0.27753479916328516, 0.16959588523287855, 0.3089119447773357, 0.2558719211959501])
    np.testing.assert_array_almost_equal(mu.values, correct_mu)
    with pytest.warns(RuntimeWarning):
        mu_np = expected_returns.capm_return(df.to_numpy())
        mu_np.name = mu.name
        mu_np.index = mu.index
        pd.testing.assert_series_equal(mu_np, mu)

def test_capm_with_benchmark():
    if False:
        print('Hello World!')
    df = get_data()
    mkt_df = get_benchmark_data()
    mu = expected_returns.capm_return(df, market_prices=mkt_df, compounding=True)
    assert isinstance(mu, pd.Series)
    assert list(mu.index) == list(df.columns)
    assert mu.notnull().all()
    assert mu.dtype == 'float64'
    correct_mu = np.array([0.09115799375654746, 0.09905386632033128, 0.05676282405265752, 0.06291827346436336, 0.13147799781014877, 0.10239088012000815, 0.1311567086884512, 0.07339649698626659, 0.1301248935078549, 0.07620949056643983, 0.07629095442513395, 0.12163575425541985, 0.10400070536161658, 0.0781736030988492, 0.09185177050469516, 0.10245700691271296, 0.11268307946677197, 0.07870087187919145, 0.1275598841214107, 0.09536788741392595])
    np.testing.assert_array_almost_equal(mu.values, correct_mu)
    mu2 = expected_returns.capm_return(df, market_prices=mkt_df, compounding=False)
    assert (mu2 >= mu).all()

def test_risk_matrix_and_returns_data():
    if False:
        while True:
            i = 10
    df = get_data()
    for method in {'mean_historical_return', 'ema_historical_return', 'capm_return'}:
        mu = expected_returns.return_model(df, method=method)
        assert isinstance(mu, pd.Series)
        assert list(mu.index) == list(df.columns)
        assert mu.notnull().all()
        assert mu.dtype == 'float64'
        mu2 = expected_returns.return_model(expected_returns.returns_from_prices(df), method=method, returns_data=True)
        pd.testing.assert_series_equal(mu, mu2)

def test_return_model_additional_kwargs():
    if False:
        return 10
    df = get_data()
    mkt_prices = get_benchmark_data()
    mu1 = expected_returns.return_model(df, method='capm_return', market_prices=mkt_prices, risk_free_rate=0.03)
    mu2 = expected_returns.capm_return(df, market_prices=mkt_prices, risk_free_rate=0.03)
    pd.testing.assert_series_equal(mu1, mu2)

def test_return_model_not_implemented():
    if False:
        i = 10
        return i + 15
    df = get_data()
    with pytest.raises(NotImplementedError):
        expected_returns.return_model(df, method='fancy_new!')

def test_log_return_passthrough():
    if False:
        while True:
            i = 10
    df = get_data()
    for method in {'mean_historical_return', 'ema_historical_return', 'capm_return'}:
        mu1 = expected_returns.return_model(df, method=method, log_returns=False)
        mu2 = expected_returns.return_model(df, method=method, log_returns=True)
        try:
            pd.testing.assert_series_equal(mu1, mu2)
        except AssertionError:
            return
        assert False