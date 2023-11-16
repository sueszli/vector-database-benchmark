import numpy as np
import pytest
from cvxpy.error import SolverError
from pypfopt import EfficientFrontier, EfficientSemivariance, expected_returns, objective_functions, risk_models
from tests.utilities_for_tests import get_data, setup_efficient_semivariance

def test_es_example():
    if False:
        i = 10
        return i + 15
    es = setup_efficient_semivariance()
    w = es.efficient_return(0.2)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    assert all([i >= -1e-05 for i in w.values()])
    np.testing.assert_allclose(es.portfolio_performance(), (0.2, 0.091287, 1.971794), rtol=0.0001, atol=0.0001)
    np.testing.assert_equal(es.portfolio_performance(verbose=True), es.portfolio_performance())

def test_es_no_returns():
    if False:
        while True:
            i = 10
    df = get_data()
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    assert EfficientSemivariance(None, historical_rets)

def test_es_return_sample():
    if False:
        i = 10
        return i + 15
    es = setup_efficient_semivariance()
    w = es.efficient_return(0.2)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    assert all([i >= -1e-05 for i in w.values()])
    np.testing.assert_allclose(es.portfolio_performance(), (0.2, 0.091287, 1.971794), rtol=0.0001, atol=0.0001)
    np.testing.assert_equal(es.portfolio_performance(verbose=True), es.portfolio_performance())

def test_es_errors():
    if False:
        return 10
    df = get_data()
    mu = expected_returns.mean_historical_return(df)
    historical_rets = expected_returns.returns_from_prices(df)
    with pytest.warns(UserWarning):
        EfficientSemivariance(mu, historical_rets)
    historical_rets = historical_rets.dropna(axis=0, how='any')
    es = EfficientSemivariance(mu, historical_rets)
    with pytest.raises(NotImplementedError):
        es.min_volatility()
    with pytest.raises(NotImplementedError):
        es.max_sharpe()
    with pytest.raises(ValueError):
        es.max_quadratic_utility(risk_aversion=-0.01)
    with pytest.raises(ValueError):
        es.efficient_return(target_return=-0.01)
    with pytest.raises(ValueError):
        es.efficient_return(target_return=np.abs(mu).max() + 0.01)
    with pytest.raises(AttributeError):
        EfficientSemivariance(mu, historical_rets.to_numpy().tolist())
    historical_rets = historical_rets.iloc[:, :-1]
    with pytest.raises(ValueError):
        EfficientSemivariance(mu, historical_rets)

def test_es_example_weekly():
    if False:
        return 10
    df = get_data()
    df = df.resample('W').first()
    mu = expected_returns.mean_historical_return(df, frequency=52)
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    es = EfficientSemivariance(mu, historical_rets, frequency=52)
    es.efficient_return(0.2)
    np.testing.assert_allclose(es.portfolio_performance(), (0.2000000562544616, 0.07667633475531543, 2.3475307841574087), rtol=0.0001, atol=0.0001)

def test_es_example_monthly():
    if False:
        print('Hello World!')
    df = get_data()
    df = df.resample('M').first()
    mu = expected_returns.mean_historical_return(df, frequency=12)
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    es = EfficientSemivariance(mu, historical_rets, frequency=12)
    es.efficient_return(0.3)
    np.testing.assert_allclose(es.portfolio_performance(), (0.3, 0.04746519522734184, 5.899059271933824), rtol=0.0001, atol=0.0001)

def test_es_example_short():
    if False:
        return 10
    df = get_data()
    mu = expected_returns.mean_historical_return(df)
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    es = EfficientSemivariance(mu, historical_rets, weight_bounds=(-1, 1))
    w = es.efficient_return(0.2, market_neutral=True)
    goog_weight = w['GOOG']
    historical_rets['GOOG'] -= historical_rets['GOOG'].quantile(0.75)
    es = EfficientSemivariance(mu, historical_rets, weight_bounds=(-1, 1))
    w = es.efficient_return(0.2, market_neutral=True)
    goog_weight2 = w['GOOG']
    assert abs(goog_weight2) >= abs(goog_weight)

def test_min_semivariance():
    if False:
        print('Hello World!')
    es = setup_efficient_semivariance()
    w = es.min_semivariance()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    assert all([i >= -1e-05 for i in w.values()])
    np.testing.assert_allclose(es.portfolio_performance(), (0.091059, 0.084974, 0.836243), rtol=0.001, atol=0.001)

def test_min_semivariance_extra_constraints():
    if False:
        return 10
    es = setup_efficient_semivariance()
    w = es.min_semivariance()
    assert w['GOOG'] < 0.02 and w['AAPL'] > 0.02
    es = setup_efficient_semivariance()
    es.add_constraint(lambda x: x[0] >= 0.03)
    es.add_constraint(lambda x: x[1] <= 0.03)
    w = es.min_semivariance()
    assert w['GOOG'] >= 0.025 and w['AAPL'] <= 0.035

def test_min_semivariance_different_solver():
    if False:
        print('Hello World!')
    es = setup_efficient_semivariance(solver='ECOS')
    w = es.min_semivariance()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    test_performance = (0.091159, 0.08496, 0.837566)
    np.testing.assert_allclose(es.portfolio_performance(), test_performance, rtol=0.01, atol=0.01)
    es = setup_efficient_semivariance(solver='OSQP')
    w = es.min_semivariance()
    np.testing.assert_allclose(es.portfolio_performance(), test_performance, rtol=0.01, atol=0.01)

def test_min_semivariance_tx_costs():
    if False:
        for i in range(10):
            print('nop')
    es = setup_efficient_semivariance()
    es.min_semivariance()
    w1 = es.weights
    es = setup_efficient_semivariance()
    prev_w = np.array([1 / es.n_assets] * es.n_assets)
    es.add_objective(objective_functions.transaction_cost, w_prev=prev_w)
    es.min_semivariance()
    w2 = es.weights
    assert np.abs(prev_w - w2).sum() < np.abs(prev_w - w1).sum()

def test_min_semivariance_L2_reg():
    if False:
        while True:
            i = 10
    es = setup_efficient_semivariance()
    es.add_objective(objective_functions.L2_reg, gamma=1)
    weights = es.min_semivariance()
    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    assert all([i >= 0 for i in weights.values()])
    ef2 = setup_efficient_semivariance()
    ef2.min_semivariance()
    equal_weight = np.full((es.n_assets,), 1 / es.n_assets)
    assert np.abs(equal_weight - es.weights).sum() < np.abs(equal_weight - ef2.weights).sum()
    np.testing.assert_allclose(es.portfolio_performance(), (0.089844, 0.112864, 0.618832), rtol=0.0001, atol=0.0001)

def test_min_semivariance_sector_constraints():
    if False:
        for i in range(10):
            print('nop')
    sector_mapper = {'GOOG': 'tech', 'AAPL': 'tech', 'FB': 'tech', 'AMZN': 'tech', 'BABA': 'tech', 'GE': 'utility', 'AMD': 'tech', 'WMT': 'retail', 'BAC': 'fig', 'GM': 'auto', 'T': 'auto', 'UAA': 'airline', 'SHLD': 'retail', 'XOM': 'energy', 'RRC': 'energy', 'BBY': 'retail', 'MA': 'fig', 'PFE': 'pharma', 'JPM': 'fig', 'SBUX': 'retail'}
    sector_upper = {'tech': 0.2, 'utility': 0.1, 'retail': 0.2, 'fig': 0.4, 'airline': 0.05, 'energy': 0.2}
    sector_lower = {'utility': 0.01, 'fig': 0.02, 'airline': 0.01}
    es = setup_efficient_semivariance()
    es.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    weights = es.min_semivariance()
    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for (t, v) in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-05
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-05

def test_max_quadratic_utility():
    if False:
        for i in range(10):
            print('nop')
    es = setup_efficient_semivariance()
    w = es.max_quadratic_utility(risk_aversion=2)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    np.testing.assert_allclose(es.portfolio_performance(), (0.50857, 0.17469, 2.796777), rtol=0.0001, atol=0.0001)

def test_max_quadratic_utility_range():
    if False:
        print('Hello World!')
    df = get_data().dropna(axis=0, how='any')
    mean_return = expected_returns.mean_historical_return(df, compounding=False)
    historic_returns = expected_returns.returns_from_prices(df)
    es = EfficientSemivariance(mean_return, historic_returns, verbose=True, solver_options={'warm_start': False})
    es.max_quadratic_utility(risk_aversion=0.01)
    (prev_ret, prev_semivar, _) = es.portfolio_performance()
    for delta in [0.1, 0.5, 1, 3, 5, 10]:
        es.max_quadratic_utility(risk_aversion=delta)
        (ret, semivar, _) = es.portfolio_performance()
        assert ret < prev_ret and semivar < prev_semivar
        prev_ret = ret
        prev_semivar = semivar

def test_max_quadratic_utility_with_shorts():
    if False:
        while True:
            i = 10
    es = setup_efficient_semivariance(weight_bounds=(-1, 1))
    es.max_quadratic_utility()
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    np.testing.assert_allclose(es.portfolio_performance(), (3.380009, 1.021973, 3.287767), rtol=0.0001, atol=0.0001)

def test_max_quadratic_utility_market_neutral():
    if False:
        i = 10
        return i + 15
    es = setup_efficient_semivariance(solver='ECOS', weight_bounds=(-1, 1))
    es.max_quadratic_utility(market_neutral=True)
    np.testing.assert_almost_equal(es.weights.sum(), 0)
    np.testing.assert_allclose(es.portfolio_performance(), (3.20978, 0.968704, 3.292832), rtol=0.0001, atol=0.0001)

def test_max_quadratic_utility_L2_reg():
    if False:
        print('Hello World!')
    es = setup_efficient_semivariance()
    es.add_objective(objective_functions.L2_reg, gamma=5)
    weights = es.max_quadratic_utility()
    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    assert all([i >= 0 for i in weights.values()])
    np.testing.assert_allclose(es.portfolio_performance(), (0.090208, 0.112854, 0.62212), rtol=0.0001, atol=0.0001)
    ef2 = setup_efficient_semivariance()
    ef2.max_quadratic_utility()
    equal_weight = np.full((es.n_assets,), 1 / es.n_assets)
    assert np.abs(equal_weight - es.weights).sum() < np.abs(equal_weight - ef2.weights).sum()

def test_efficient_risk():
    if False:
        print('Hello World!')
    es = setup_efficient_semivariance()
    w = es.efficient_risk(0.2)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    assert all([i >= -1e-05 for i in w.values()])
    np.testing.assert_allclose(es.portfolio_performance(), (0.508571, 0.174691, 2.796777), rtol=0.0001, atol=0.0001)

def test_efficient_risk_low_risk():
    if False:
        i = 10
        return i + 15
    es = setup_efficient_semivariance()
    es.min_semivariance()
    min_value = es.portfolio_performance()[1]
    with pytest.raises(SolverError):
        es = setup_efficient_semivariance()
        es.efficient_risk(min_value - 0.01)
    es = setup_efficient_semivariance()
    es.efficient_risk(min_value + 0.01)
    np.testing.assert_allclose(es.portfolio_performance(), (0.228226, min_value + 0.01, 2.192011), rtol=0.0001, atol=0.0001)

def test_efficient_risk_market_neutral():
    if False:
        i = 10
        return i + 15
    es = setup_efficient_semivariance(weight_bounds=(-1, 1))
    w = es.efficient_risk(0.21, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 0)
    assert (es.weights < 1).all() and (es.weights > -1).all()
    np.testing.assert_allclose(es.portfolio_performance(), (1.020958, 0.210008, 4.766278), rtol=0.0001, atol=0.0001)

def test_efficient_risk_L2_reg():
    if False:
        return 10
    es = setup_efficient_semivariance()
    es.add_objective(objective_functions.L2_reg, gamma=1)
    weights = es.efficient_risk(0.19)
    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    np.testing.assert_array_less(np.zeros(len(weights)), es.weights + 0.0001)
    np.testing.assert_allclose(es.portfolio_performance(), (0.288996, 0.131377, 2.047509), rtol=0.0001, atol=0.0001)
    ef2 = setup_efficient_semivariance()
    ef2.efficient_risk(0.19)
    equal_weight = np.full((es.n_assets,), 1 / es.n_assets)
    assert np.abs(equal_weight - es.weights).sum() < np.abs(equal_weight - ef2.weights).sum()

def test_efficient_return():
    if False:
        return 10
    es = setup_efficient_semivariance()
    w = es.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    assert all([i >= -1e-05 for i in w.values()])
    np.testing.assert_allclose(es.portfolio_performance(), (0.25, 0.098453, 2.33615), rtol=0.0001, atol=0.0001)

def test_efficient_return_short():
    if False:
        for i in range(10):
            print('nop')
    es = setup_efficient_semivariance(weight_bounds=(None, None))
    w = es.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    np.testing.assert_allclose(es.portfolio_performance(), (0.25, 0.090188, 2.550234), rtol=0.0001, atol=0.0001)
    sortino = es.portfolio_performance()[2]
    ef_long_only = setup_efficient_semivariance()
    ef_long_only.efficient_return(0.25)
    long_only_sortino = ef_long_only.portfolio_performance()[2]
    assert sortino > long_only_sortino

def test_efficient_return_L2_reg():
    if False:
        for i in range(10):
            print('nop')
    es = setup_efficient_semivariance()
    es.add_objective(objective_functions.L2_reg, gamma=1)
    w = es.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(es.portfolio_performance(), (0.25, 0.121407, 1.894448), rtol=0.0001, atol=0.0001)

def test_efficient_semivariance_vs_heuristic():
    if False:
        for i in range(10):
            print('nop')
    benchmark = 0
    es = setup_efficient_semivariance()
    es.efficient_return(0.2)
    (mu_es, semi_deviation, _) = es.portfolio_performance()
    np.testing.assert_almost_equal(mu_es, 0.2)
    (mean_return, historic_returns) = setup_efficient_semivariance(data_only=True)
    pairwise_semivariance = risk_models.semicovariance(historic_returns, returns_data=True, benchmark=0, frequency=1)
    ef = EfficientFrontier(mean_return, pairwise_semivariance)
    ef.efficient_return(0.2)
    (mu_ef, _, _) = ef.portfolio_performance()
    portfolio_returns = historic_returns @ ef.weights
    drops = np.fmin(portfolio_returns - benchmark, 0)
    T = historic_returns.shape[0]
    semivariance = np.sum(np.square(drops)) / T * 252
    semi_deviation_ef = np.sqrt(semivariance)
    assert semi_deviation < semi_deviation_ef
    assert mu_es / semi_deviation > mu_ef / semi_deviation_ef

def test_efficient_semivariance_vs_heuristic_weekly():
    if False:
        while True:
            i = 10
    benchmark = 0
    (_, historic_returns) = setup_efficient_semivariance(data_only=True)
    weekly_returns = historic_returns.resample('W').sum()
    mean_weekly_returns = weekly_returns.mean(axis=0)
    es = EfficientSemivariance(mean_weekly_returns, weekly_returns, frequency=52)
    es.efficient_return(0.2 / 52)
    (mu_es, semi_deviation, _) = es.portfolio_performance()
    pairwise_semivariance = risk_models.semicovariance(weekly_returns, returns_data=True, benchmark=0, frequency=1)
    ef = EfficientFrontier(mean_weekly_returns, pairwise_semivariance)
    ef.efficient_return(0.2 / 52)
    (mu_ef, _, _) = ef.portfolio_performance()
    portfolio_returns = historic_returns @ ef.weights
    drops = np.fmin(portfolio_returns - benchmark, 0)
    T = weekly_returns.shape[0]
    semivariance = np.sum(np.square(drops)) / T * 52
    semi_deviation_ef = np.sqrt(semivariance)
    assert semi_deviation < semi_deviation_ef
    assert mu_es / semi_deviation > mu_ef / semi_deviation_ef

def test_parametrization():
    if False:
        while True:
            i = 10
    es = setup_efficient_semivariance()
    es.efficient_risk(0.19)
    es.efficient_risk(0.19)
    es = setup_efficient_semivariance()
    es.efficient_return(0.25)
    es.efficient_return(0.25)
    es = setup_efficient_semivariance()
    es.max_quadratic_utility(1)
    es.max_quadratic_utility(1)