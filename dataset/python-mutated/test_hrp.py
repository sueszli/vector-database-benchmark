import numpy as np
import pandas as pd
import pytest
from pypfopt import CovarianceShrinkage, HRPOpt
from tests.utilities_for_tests import get_data, resource

def test_hrp_errors():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        hrp = HRPOpt()
    df = get_data()
    returns = df.pct_change().dropna(how='all')
    returns_np = returns.to_numpy()
    with pytest.raises(TypeError):
        hrp = HRPOpt(returns_np)
    hrp = HRPOpt(returns)
    with pytest.raises(ValueError):
        hrp.optimize(linkage_method='blah')

def test_hrp_portfolio():
    if False:
        while True:
            i = 10
    df = get_data()
    returns = df.pct_change().dropna(how='all')
    hrp = HRPOpt(returns)
    w = hrp.optimize(linkage_method='single')
    x = pd.read_csv(resource('weights_hrp.csv'), index_col=0).squeeze('columns')
    pd.testing.assert_series_equal(x, pd.Series(w), check_names=False, rtol=0.01)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(df.columns)
    np.testing.assert_almost_equal(sum(w.values()), 1)
    assert all([i >= 0 for i in w.values()])

def test_portfolio_performance():
    if False:
        return 10
    df = get_data()
    returns = df.pct_change().dropna(how='all')
    hrp = HRPOpt(returns)
    with pytest.raises(ValueError):
        hrp.portfolio_performance()
    hrp.optimize(linkage_method='single')
    np.testing.assert_allclose(hrp.portfolio_performance(), (0.21353402380950973, 0.17844159743748936, 1.084579081272277))

def test_pass_cov_matrix():
    if False:
        i = 10
        return i + 15
    df = get_data()
    S = CovarianceShrinkage(df).ledoit_wolf()
    hrp = HRPOpt(cov_matrix=S)
    hrp.optimize(linkage_method='single')
    perf = hrp.portfolio_performance()
    assert perf[0] is None and perf[2] is None
    np.testing.assert_almost_equal(perf[1], 0.10002783894982334)

def test_cluster_var():
    if False:
        for i in range(10):
            print('nop')
    df = get_data()
    returns = df.pct_change().dropna(how='all')
    cov = returns.cov()
    tickers = ['SHLD', 'AMD', 'BBY', 'RRC', 'FB', 'WMT', 'T', 'BABA', 'PFE', 'UAA']
    var = HRPOpt._get_cluster_var(cov, tickers)
    np.testing.assert_almost_equal(var, 0.00012842967106653283)

def test_quasi_dag():
    if False:
        i = 10
        return i + 15
    df = get_data()
    returns = df.pct_change().dropna(how='all')
    hrp = HRPOpt(returns)
    hrp.optimize(linkage_method='single')
    clusters = hrp.clusters
    assert HRPOpt._get_quasi_diag(clusters)[:5] == [12, 6, 15, 14, 2]