import numpy as np
import pandas as pd
import pytest
from pypfopt import expected_returns, risk_models
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns, market_implied_risk_aversion
from tests.utilities_for_tests import get_data, get_market_caps, resource

def test_input_errors():
    if False:
        return 10
    df = get_data()
    S = risk_models.sample_cov(df)
    views = pd.Series(0.1, index=S.columns)
    with pytest.raises(TypeError):
        BlackLittermanModel(S)
    assert BlackLittermanModel(S, Q=views)
    with pytest.raises(ValueError):
        BlackLittermanModel(S, Q=views, tau=-0.1)
    P = np.eye(len(S))[:, :-1]
    with pytest.raises(AssertionError):
        BlackLittermanModel(S, Q=views, P=P)
    with pytest.raises(TypeError):
        BlackLittermanModel(S, Q=views[:-1], P=1.0)
    with pytest.raises(AssertionError):
        BlackLittermanModel(S, Q=views, P=P, omega=np.eye(len(views)))
    with pytest.raises(AssertionError):
        BlackLittermanModel(S, Q=views, pi=df.pct_change().mean()[:-1])
    with pytest.raises(ValueError):
        BlackLittermanModel(S, Q=views, pi='market')
    with pytest.raises(TypeError):
        BlackLittermanModel(S, Q=views, pi=[0.1] * len(S))
    with pytest.raises(ValueError):
        BlackLittermanModel(S, Q=views, risk_aversion=-0.01)
    with pytest.raises(TypeError):
        BlackLittermanModel(S, Q=views, omega=1.0)

def test_parse_views():
    if False:
        return 10
    df = get_data()
    S = risk_models.sample_cov(df)
    viewlist = ['AAPL', 0.2, 'GOOG', -0.3, 'XOM', 0.4]
    viewdict = {'AAPL': 0.2, 'GOOG': -0.3, 'XOM': 0.4, 'fail': 0.1}
    with pytest.raises(TypeError):
        bl = BlackLittermanModel(S, absolute_views=viewlist)
    with pytest.raises(ValueError):
        bl = BlackLittermanModel(S, absolute_views=viewdict)
    del viewdict['fail']
    bl = BlackLittermanModel(S, absolute_views=viewdict)
    test_P = np.copy(bl.P)
    test_P[0, 1] -= 1
    test_P[1, 0] -= 1
    test_P[2, 13] -= 1
    np.testing.assert_array_equal(test_P, np.zeros((len(bl.Q), bl.n_assets)))
    np.testing.assert_array_equal(bl.Q, np.array([0.2, -0.3, 0.4]).reshape(-1, 1))

def test_dataframe_input():
    if False:
        for i in range(10):
            print('nop')
    df = get_data()
    S = risk_models.sample_cov(df)
    view_df = pd.DataFrame(pd.Series(0.1, index=S.columns))
    bl = BlackLittermanModel(S, Q=view_df)
    np.testing.assert_array_equal(bl.P, np.eye(len(view_df)))
    view_df = pd.DataFrame(pd.Series(0.1, index=S.columns)[:10])
    picking = pd.DataFrame(np.eye(len(S))[:10, :])
    assert BlackLittermanModel(S, Q=view_df, P=picking)
    prior_df = df.pct_change().mean()
    assert BlackLittermanModel(S, pi=prior_df, Q=view_df, P=picking)
    omega_df = S.iloc[:10, :10]
    assert BlackLittermanModel(S, pi=prior_df, Q=view_df, P=picking, omega=omega_df)

def test_cov_ndarray():
    if False:
        return 10
    df = get_data()
    prior_df = df.pct_change().mean()
    S = risk_models.sample_cov(df)
    views = pd.Series(0.1, index=S.columns)
    bl = BlackLittermanModel(S, pi=prior_df, Q=views)
    bl_nd = BlackLittermanModel(S.to_numpy(), pi=prior_df.to_numpy(), Q=views)
    np.testing.assert_equal(bl_nd.bl_returns().to_numpy(), bl.bl_returns().to_numpy())
    np.testing.assert_equal(bl_nd.bl_cov().to_numpy(), bl.bl_cov().to_numpy())
    assert list(bl_nd.bl_weights().values()) == list(bl.bl_weights().values())

def test_default_omega():
    if False:
        return 10
    df = get_data()
    S = risk_models.sample_cov(df)
    views = pd.Series(0.1, index=S.columns)
    bl = BlackLittermanModel(S, Q=views)
    assert bl.omega.shape == (len(S), len(S))
    np.testing.assert_array_equal(bl.omega, np.diag(np.diagonal(bl.omega)))
    np.testing.assert_array_almost_equal(np.diagonal(bl.omega), bl.tau * np.diagonal(S))

def test_bl_returns_no_prior():
    if False:
        while True:
            i = 10
    df = get_data()
    S = risk_models.sample_cov(df)
    viewdict = {'AAPL': 0.2, 'BBY': -0.3, 'BAC': 0, 'SBUX': -0.2, 'T': 0.131321}
    bl = BlackLittermanModel(S, absolute_views=viewdict)
    rets = bl.bl_returns()
    test_rets = np.linalg.inv(np.linalg.inv(bl.tau * bl.cov_matrix) + bl.P.T @ np.linalg.inv(bl.omega) @ bl.P) @ (bl.P.T @ np.linalg.inv(bl.omega) @ bl.Q)
    np.testing.assert_array_almost_equal(rets.values.reshape(-1, 1), test_rets)

def test_bl_equal_prior():
    if False:
        i = 10
        return i + 15
    df = get_data()
    S = risk_models.sample_cov(df)
    viewdict = {'AAPL': 0.2, 'BBY': -0.3, 'BAC': 0, 'SBUX': -0.2, 'T': 0.131321}
    bl = BlackLittermanModel(S, absolute_views=viewdict, pi='equal')
    np.testing.assert_array_almost_equal(bl.pi, np.ones((20, 1)) * 0.05)
    bl.bl_weights()
    np.testing.assert_allclose(bl.portfolio_performance(), (0.1877432247395778, 0.3246889329226965, 0.5166274785827545))

def test_bl_returns_all_views():
    if False:
        return 10
    df = get_data()
    prior = expected_returns.ema_historical_return(df)
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    views = pd.Series(0.1, index=S.columns)
    bl = BlackLittermanModel(S, pi=prior, Q=views)
    posterior_rets = bl.bl_returns()
    assert isinstance(posterior_rets, pd.Series)
    assert list(posterior_rets.index) == list(df.columns)
    assert posterior_rets.notnull().all()
    assert posterior_rets.dtype == 'float64'
    np.testing.assert_array_almost_equal(posterior_rets, np.array([0.11168648, 0.16782938, 0.12516799, 0.24067997, 0.32848296, -0.22789895, 0.16311297, 0.11928542, 0.25414308, 0.11007738, 0.06282615, -0.03140218, -0.16977172, 0.05254821, -0.10463884, 0.32173375, 0.26399864, 0.1118594, 0.22999558, 0.08977448]))

def test_bl_relative_views():
    if False:
        i = 10
        return i + 15
    df = get_data()
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    views = np.array([-0.2, 0.1, 0.15]).reshape(-1, 1)
    picking = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0]])
    bl = BlackLittermanModel(S, Q=views, P=picking)
    rets = bl.bl_returns()
    assert rets['SBUX'] < 0
    assert rets['GOOG'] > rets['FB']
    assert rets['BAC'] > rets['T'] and rets['JPM'] > rets['GE']

def test_bl_cov_default():
    if False:
        while True:
            i = 10
    df = get_data()
    cov_matrix = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    viewdict = {'AAPL': 0.2, 'BBY': -0.3, 'BAC': 0, 'SBUX': -0.2, 'T': 0.131321}
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)
    S = bl.bl_cov()
    assert S.shape == (20, 20)
    assert S.index.equals(df.columns)
    assert S.index.equals(S.columns)
    assert S.notnull().all().all()

def test_market_risk_aversion():
    if False:
        i = 10
        return i + 15
    prices = pd.read_csv(resource('spy_prices.csv'), parse_dates=True, index_col=0).squeeze('columns')
    delta = market_implied_risk_aversion(prices)
    assert np.round(delta, 5) == 2.68549
    prices = pd.read_csv(resource('spy_prices.csv'), parse_dates=True, index_col=0)
    delta = market_implied_risk_aversion(prices)
    assert np.round(delta.iloc[0], 5) == 2.68549
    list_invalid = [100.0, 110.0, 120.0, 130.0]
    with pytest.raises(TypeError):
        delta = market_implied_risk_aversion(list_invalid)

def test_bl_weights():
    if False:
        print('Hello World!')
    df = get_data()
    S = risk_models.sample_cov(df)
    viewdict = {'AAPL': 0.2, 'BBY': -0.3, 'BAC': 0, 'SBUX': -0.2, 'T': 0.131321}
    bl = BlackLittermanModel(S, absolute_views=viewdict)
    prices = pd.read_csv(resource('spy_prices.csv'), parse_dates=True, index_col=0).squeeze('columns')
    delta = market_implied_risk_aversion(prices)
    bl.bl_weights(delta)
    w = bl.clean_weights()
    assert abs(sum(w.values()) - 1) < 1e-05
    assert all((viewdict[t] * w[t] >= 0 for t in viewdict))
    test_weights = {'GOOG': 0.0, 'AAPL': 1.40675, 'FB': 0.0, 'BABA': 0.0, 'AMZN': 0.0, 'GE': 0.0, 'AMD': 0.0, 'WMT': 0.0, 'BAC': 0.02651, 'GM': 0.0, 'T': 2.81117, 'UAA': 0.0, 'SHLD': 0.0, 'XOM': 0.0, 'RRC': 0.0, 'BBY': -1.44667, 'MA': 0.0, 'PFE': 0.0, 'JPM': 0.0, 'SBUX': -1.79776}
    assert w == test_weights
    bl = BlackLittermanModel(S, absolute_views=viewdict)
    bl.optimize(delta)
    w2 = bl.clean_weights()
    assert w2 == w
    bl = BlackLittermanModel(S, absolute_views=pd.Series(viewdict))
    bl.optimize(delta)
    w2 = bl.clean_weights()
    assert w2 == w

def test_market_implied_prior():
    if False:
        return 10
    df = get_data()
    S = risk_models.sample_cov(df)
    prices = pd.read_csv(resource('spy_prices.csv'), parse_dates=True, index_col=0).squeeze('columns')
    delta = market_implied_risk_aversion(prices)
    mcaps = get_market_caps()
    pi = market_implied_prior_returns(mcaps, delta, S)
    assert isinstance(pi, pd.Series)
    assert list(pi.index) == list(df.columns)
    assert pi.notnull().all()
    assert pi.dtype == 'float64'
    np.testing.assert_array_almost_equal(pi.values, np.array([0.14933293, 0.2168623, 0.11219185, 0.10362374, 0.28416295, 0.12196098, 0.19036819, 0.08860159, 0.17724273, 0.08779627, 0.0791797, 0.16460474, 0.12854665, 0.08657863, 0.11230036, 0.13875465, 0.15017163, 0.09066484, 0.1696369, 0.13270213]))
    mcaps = pd.Series(mcaps)
    pi2 = market_implied_prior_returns(mcaps, delta, S)
    pd.testing.assert_series_equal(pi, pi2, check_exact=False)
    bl = BlackLittermanModel(S, pi='market', market_caps=mcaps, absolute_views={'AAPL': 0.1}, risk_aversion=delta)
    pi = market_implied_prior_returns(mcaps, delta, S, risk_free_rate=0)
    np.testing.assert_array_almost_equal(bl.pi, pi.values.reshape(-1, 1))

def test_bl_market_prior():
    if False:
        while True:
            i = 10
    df = get_data()
    S = risk_models.sample_cov(df)
    prices = pd.read_csv(resource('spy_prices.csv'), parse_dates=True, index_col=0).squeeze('columns')
    delta = market_implied_risk_aversion(prices)
    mcaps = get_market_caps()
    with pytest.warns(RuntimeWarning):
        market_implied_prior_returns(mcaps, delta, S.values)
    prior = market_implied_prior_returns(mcaps, delta, S)
    viewdict = {'GOOG': 0.4, 'AAPL': -0.3, 'FB': 0.3, 'BABA': 0}
    bl = BlackLittermanModel(S, pi=prior, absolute_views=viewdict)
    rets = bl.bl_returns()
    for v in viewdict:
        assert prior[v] <= rets[v] <= viewdict[v] or viewdict[v] <= rets[v] <= prior[v]
    with pytest.raises(ValueError):
        bl.portfolio_performance()
    bl.bl_weights(delta)
    np.testing.assert_allclose(bl.portfolio_performance(), (0.2580693114409672, 0.265445955488424, 0.8968654692926723))
    assert bl.posterior_cov is not None

def test_bl_market_automatic():
    if False:
        while True:
            i = 10
    df = get_data()
    S = risk_models.sample_cov(df)
    mcaps = get_market_caps()
    viewdict = {'GOOG': 0.4, 'AAPL': -0.3, 'FB': 0.3, 'BABA': 0}
    bl = BlackLittermanModel(S, pi='market', absolute_views=viewdict, market_caps=mcaps)
    rets = bl.bl_returns()
    prior = market_implied_prior_returns(mcaps, 1, S, 0)
    bl2 = BlackLittermanModel(S, pi=prior, absolute_views=viewdict)
    rets2 = bl2.bl_returns()
    pd.testing.assert_series_equal(rets, rets2)

def test_bl_tau():
    if False:
        i = 10
        return i + 15
    df = get_data()
    S = risk_models.sample_cov(df)
    prices = pd.read_csv(resource('spy_prices.csv'), parse_dates=True, index_col=0).squeeze('columns')
    delta = market_implied_risk_aversion(prices)
    mcaps = get_market_caps()
    prior = market_implied_prior_returns(mcaps, delta, S)
    viewdict = {'GOOG': 0.4, 'AAPL': -0.3, 'FB': 0.3, 'BABA': 0}
    omega = np.diag([0.01, 0.01, 0.01, 0.01])
    bl0 = BlackLittermanModel(S, pi=prior, absolute_views=viewdict, tau=1e-10, omega=omega)
    bl1 = BlackLittermanModel(S, pi=prior, absolute_views=viewdict, tau=0.01, omega=omega)
    bl2 = BlackLittermanModel(S, pi=prior, absolute_views=viewdict, tau=0.1, omega=omega)
    np.testing.assert_allclose(bl0.bl_returns(), bl0.pi.flatten(), rtol=1e-05)
    assert bl1.bl_returns()['GOOG'] > bl0.bl_returns()['GOOG']
    assert bl2.bl_returns()['GOOG'] > bl1.bl_returns()['GOOG']

def test_bl_no_uncertainty():
    if False:
        print('Hello World!')
    df = get_data()
    S = risk_models.sample_cov(df)
    omega = np.diag([0, 0, 0, 0])
    viewdict = {'GOOG': 0.4, 'AAPL': -0.3, 'FB': 0.3, 'BABA': 0}
    bl = BlackLittermanModel(S, absolute_views=viewdict, omega=omega)
    rets = bl.bl_returns()
    for (k, v) in viewdict.items():
        assert np.abs(rets[k] - v) < 1e-05
    omega = np.diag([0, 0.2, 0.2, 0.2])
    bl = BlackLittermanModel(S, absolute_views=viewdict, omega=omega)
    rets = bl.bl_returns()
    assert np.abs(bl.bl_returns()['GOOG'] - viewdict['GOOG']) < 1e-05
    assert np.abs(rets['AAPL'] - viewdict['AAPL']) > 0.01

def test_idzorek_confidences_error():
    if False:
        for i in range(10):
            print('nop')
    S = pd.DataFrame(np.diag(np.ones((5,))), index=range(5), columns=range(5))
    views = {k: 0.3 for k in range(5)}
    pi = pd.Series(0.1, index=range(5))
    with pytest.raises(ValueError):
        BlackLittermanModel(S, pi=pi, absolute_views=views, omega='idzorek')
    with pytest.raises(ValueError):
        BlackLittermanModel(S, pi=pi, absolute_views=views, omega='idzorek', view_confidences=[0.2] * 4)
    with pytest.raises(ValueError):
        BlackLittermanModel(S, pi=pi, absolute_views=views, omega='idzorek', view_confidences=[1.1] * 5)
    with pytest.raises(ValueError):
        BlackLittermanModel(S, pi=pi, absolute_views=views, omega='idzorek', view_confidences=[-0.1] * 5)

def test_idzorek_basic():
    if False:
        while True:
            i = 10
    S = pd.DataFrame(np.diag(np.ones((5,))), index=range(5), columns=range(5))
    views = {k: 0.3 for k in range(5)}
    pi = pd.Series(0.1, index=range(5))
    bl = BlackLittermanModel(S, pi=pi, absolute_views=views, omega=np.diag(np.zeros(5)))
    pd.testing.assert_series_equal(bl.bl_returns(), pd.Series([0.3] * 5))
    bl = BlackLittermanModel(S, pi=pi, absolute_views=views, omega=S * 1000000.0)
    pd.testing.assert_series_equal(bl.bl_returns(), pi)
    bl = BlackLittermanModel(S, pi=pi, absolute_views=views, omega='idzorek', view_confidences=[1] * 5)
    np.testing.assert_array_almost_equal(bl.omega, np.zeros((5, 5)))
    pd.testing.assert_series_equal(bl.bl_returns(), pd.Series(0.3, index=range(5)))
    bl = BlackLittermanModel(S, pi=pi, absolute_views=views, omega='idzorek', view_confidences=[0] * 5)
    np.testing.assert_array_almost_equal(bl.omega, np.diag([1000000.0] * 5))
    pd.testing.assert_series_equal(bl.bl_returns(), pi)
    for (i, conf) in enumerate(np.arange(0, 1.2, 0.2)):
        bl = BlackLittermanModel(S, pi=pi, absolute_views=views, omega='idzorek', view_confidences=[conf] * 5)
        np.testing.assert_almost_equal(bl.bl_returns()[0], 0.1 + i * 0.2 / 5)

def test_idzorek_input_formats():
    if False:
        return 10
    S = pd.DataFrame(np.diag(np.ones((5,))), index=range(5), columns=range(5))
    views = {k: 0.3 for k in range(5)}
    pi = pd.Series(0.1, index=range(5))
    test_result = pd.Series(0.2, index=range(5))
    bl = BlackLittermanModel(S, pi=pi, absolute_views=views, omega='idzorek', view_confidences=[0.5] * 5)
    pd.testing.assert_series_equal(bl.bl_returns(), test_result)
    bl = BlackLittermanModel(S, pi=pi, absolute_views=views, omega='idzorek', view_confidences=(0.5, 0.5, 0.5, 0.5, 0.5))
    pd.testing.assert_series_equal(bl.bl_returns(), test_result)
    bl = BlackLittermanModel(S, pi=pi, absolute_views=views, omega='idzorek', view_confidences=np.array([0.5] * 5))
    pd.testing.assert_series_equal(bl.bl_returns(), test_result)
    bl = BlackLittermanModel(S, pi=pi, absolute_views=views, omega='idzorek', view_confidences=np.array([0.5] * 5).reshape(-1, 1))
    pd.testing.assert_series_equal(bl.bl_returns(), test_result)

def test_idzorek_with_priors():
    if False:
        return 10
    df = get_data()
    S = risk_models.sample_cov(df)
    mcaps = get_market_caps()
    viewdict = {'GOOG': 0.4, 'AAPL': -0.3, 'FB': 0.3, 'BABA': 0}
    bl = BlackLittermanModel(S, pi='market', market_caps=mcaps, absolute_views=viewdict, omega='idzorek', view_confidences=[1, 1, 0.25, 0.25])
    rets = bl.bl_returns()
    assert bl.omega[0, 0] == 0
    np.testing.assert_almost_equal(rets['AAPL'], -0.3)
    with pytest.raises(ValueError):
        bl.portfolio_performance()
    bl.bl_weights()
    np.testing.assert_allclose(bl.portfolio_performance(), (0.943431295405105, 0.5361412623208567, 1.722365653051476))
    assert bl.posterior_cov is not None