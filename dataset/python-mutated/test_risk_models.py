import numpy as np
import pandas as pd
import pytest
from pypfopt import expected_returns, risk_models
from tests.utilities_for_tests import get_data

def test_sample_cov_dummy():
    if False:
        while True:
            i = 10
    data = pd.DataFrame([[4.0, 2.0, 0.6], [4.2, 2.1, 0.59], [3.9, 2.0, 0.58], [4.3, 2.1, 0.62], [4.1, 2.2, 0.63]])
    test_answer = pd.DataFrame([[0.006661687937656102, 0.00264970955585574, 0.0020849735375206195], [0.00264970955585574, 0.0023450491307634215, 0.00096770864287974], [0.0020849735375206195, 0.00096770864287974, 0.0016396416271856837]])
    S = risk_models.sample_cov(data) / 252
    pd.testing.assert_frame_equal(S, test_answer)

def test_sample_cov_log_dummy():
    if False:
        for i in range(10):
            print('nop')
    data = pd.DataFrame([[4.0, 2.0, 0.6], [4.2, 2.1, 0.59], [3.9, 2.0, 0.58], [4.3, 2.1, 0.62], [4.1, 2.2, 0.63]])
    test_answer = pd.DataFrame([[0.006507, 0.002652, 0.001965], [0.002652, 0.002345, 0.00095], [0.001965, 0.00095, 0.001561]])
    S = risk_models.sample_cov(data, log_returns=True) / 252
    pd.testing.assert_frame_equal(S, test_answer, atol=1e-05)

def test_is_positive_semidefinite():
    if False:
        return 10
    a = np.zeros((100, 100))
    assert risk_models._is_positive_semidefinite(a)

def test_sample_cov_real_data():
    if False:
        i = 10
        return i + 15
    df = get_data()
    S = risk_models.sample_cov(df)
    assert S.shape == (20, 20)
    assert S.index.equals(df.columns)
    assert S.index.equals(S.columns)
    assert S.notnull().all().all()
    assert risk_models._is_positive_semidefinite(S)

def test_sample_cov_type_warning():
    if False:
        print('Hello World!')
    df = get_data()
    cov_from_df = risk_models.sample_cov(df)
    returns_as_array = np.array(df)
    with pytest.warns(RuntimeWarning) as w:
        cov_from_array = risk_models.sample_cov(returns_as_array)
        assert len(w) == 1
        assert str(w[0].message) == 'data is not in a dataframe'
    np.testing.assert_array_almost_equal(cov_from_df.values, cov_from_array.values, decimal=6)

def test_sample_cov_npd():
    if False:
        print('Hello World!')
    S = np.array([[0.03818144, 0.04182824], [0.04182824, 0.04149209]])
    assert not risk_models._is_positive_semidefinite(S)
    for method in {'spectral', 'diag'}:
        with pytest.warns(UserWarning) as w:
            S2 = risk_models.fix_nonpositive_semidefinite(S, fix_method=method)
            assert risk_models._is_positive_semidefinite(S2)
            assert len(w) == 1
            assert str(w[0].message) == 'The covariance matrix is non positive semidefinite. Amending eigenvalues.'
            tickers = ['A', 'B']
            S_df = pd.DataFrame(data=S, index=tickers, columns=tickers)
            S2_df = risk_models.fix_nonpositive_semidefinite(S_df, fix_method=method)
            assert isinstance(S2_df, pd.DataFrame)
            np.testing.assert_equal(S2_df.to_numpy(), S2)
            assert S2_df.index.equals(S_df.index)
            assert S2_df.columns.equals(S_df.columns)
    with pytest.warns(UserWarning):
        with pytest.raises(NotImplementedError):
            risk_models.fix_nonpositive_semidefinite(S, fix_method='blah')

def test_fix_npd_different_method():
    if False:
        for i in range(10):
            print('nop')
    df = get_data()
    S = risk_models.sample_cov(df)
    assert risk_models._is_positive_semidefinite(S)
    S = risk_models.sample_cov(df, fix_method='diag')
    assert risk_models._is_positive_semidefinite(S)

def test_sample_cov_frequency():
    if False:
        i = 10
        return i + 15
    df = get_data()
    S = risk_models.sample_cov(df)
    S2 = risk_models.sample_cov(df, frequency=2)
    pd.testing.assert_frame_equal(S / 126, S2)

def test_semicovariance():
    if False:
        i = 10
        return i + 15
    df = get_data()
    S = risk_models.semicovariance(df)
    assert S.shape == (20, 20)
    assert S.index.equals(df.columns)
    assert S.index.equals(S.columns)
    assert S.notnull().all().all()
    assert risk_models._is_positive_semidefinite(S)
    S2 = risk_models.semicovariance(df, frequency=2)
    pd.testing.assert_frame_equal(S / 126, S2)
    with pytest.warns(RuntimeWarning):
        S2_np = risk_models.semicovariance(df.to_numpy(), frequency=2)
        np.testing.assert_equal(S2_np, S2.to_numpy())

def test_semicovariance_benchmark():
    if False:
        print('Hello World!')
    df = get_data()
    S_negative_benchmark = risk_models.semicovariance(df, benchmark=-0.5)
    np.testing.assert_allclose(S_negative_benchmark, 0, atol=0.0001)
    S = risk_models.semicovariance(df, benchmark=0)
    S2 = risk_models.semicovariance(df, benchmark=1)
    assert S2.sum().sum() > S.sum().sum()

def test_exp_cov_matrix():
    if False:
        while True:
            i = 10
    df = get_data()
    S = risk_models.exp_cov(df)
    assert S.shape == (20, 20)
    assert S.index.equals(df.columns)
    assert S.index.equals(S.columns)
    assert S.notnull().all().all()
    assert risk_models._is_positive_semidefinite(S)
    S2 = risk_models.exp_cov(df, frequency=2)
    pd.testing.assert_frame_equal(S / 126, S2)
    with pytest.warns(RuntimeWarning):
        S2_np = risk_models.exp_cov(df.to_numpy(), frequency=2)
        np.testing.assert_equal(S2_np, S2.to_numpy())
    with pytest.warns(UserWarning):
        risk_models.exp_cov(df, frequency=2, span=9)

def test_exp_cov_limits():
    if False:
        i = 10
        return i + 15
    df = get_data()
    sample_cov = risk_models.sample_cov(df)
    S = risk_models.exp_cov(df)
    assert not np.allclose(sample_cov, S)
    S2 = risk_models.exp_cov(df, span=1e+20)
    assert np.abs(S2 - sample_cov).max().max() < 0.001

def test_cov_to_corr():
    if False:
        for i in range(10):
            print('nop')
    df = get_data()
    rets = risk_models.returns_from_prices(df).dropna()
    test_corr = risk_models.cov_to_corr(rets.cov())
    pd.testing.assert_frame_equal(test_corr, rets.corr())
    with pytest.warns(RuntimeWarning) as w:
        test_corr_numpy = risk_models.cov_to_corr(rets.cov().values)
        assert len(w) == 1
        assert str(w[0].message) == 'cov_matrix is not a dataframe'
        assert isinstance(test_corr_numpy, pd.DataFrame)
        np.testing.assert_array_almost_equal(test_corr_numpy, rets.corr().values)

def test_corr_to_cov():
    if False:
        i = 10
        return i + 15
    df = get_data()
    rets = risk_models.returns_from_prices(df).dropna()
    test_corr = risk_models.cov_to_corr(rets.cov())
    new_cov = risk_models.corr_to_cov(test_corr, rets.std())
    pd.testing.assert_frame_equal(new_cov, rets.cov())
    with pytest.warns(RuntimeWarning) as w:
        cov_numpy = risk_models.corr_to_cov(test_corr.to_numpy(), rets.std())
        assert len(w) == 1
        assert str(w[0].message) == 'corr_matrix is not a dataframe'
        assert isinstance(cov_numpy, pd.DataFrame)
        np.testing.assert_equal(cov_numpy.to_numpy(), new_cov.to_numpy())

def test_covariance_shrinkage_init():
    if False:
        while True:
            i = 10
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    assert cs.S.shape == (20, 20)
    assert not np.isnan(cs.S).any()

def test_shrunk_covariance():
    if False:
        return 10
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    shrunk_cov = cs.shrunk_covariance(0.2)
    assert cs.delta == 0.2
    assert shrunk_cov.shape == (20, 20)
    assert list(shrunk_cov.index) == list(df.columns)
    assert list(shrunk_cov.columns) == list(df.columns)
    assert not shrunk_cov.isnull().any().any()
    assert risk_models._is_positive_semidefinite(shrunk_cov)
    with pytest.warns(RuntimeWarning) as w:
        cs_numpy = risk_models.CovarianceShrinkage(df.to_numpy())
        assert len(w) == 1
        assert str(w[0].message) == 'data is not in a dataframe'
        shrunk_cov_numpy = cs_numpy.shrunk_covariance(0.2)
        assert isinstance(shrunk_cov_numpy, pd.DataFrame)
        np.testing.assert_equal(shrunk_cov_numpy.to_numpy(), shrunk_cov.to_numpy())

def test_shrunk_covariance_extreme_delta():
    if False:
        print('Hello World!')
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    shrunk_cov = cs.shrunk_covariance(0)
    np.testing.assert_array_almost_equal(shrunk_cov.values, risk_models.sample_cov(df))
    shrunk_cov = cs.shrunk_covariance(1)
    N = df.shape[1]
    F = np.identity(N) * np.trace(cs.S) / N
    np.testing.assert_array_almost_equal(shrunk_cov.values, F * 252)

def test_shrunk_covariance_frequency():
    if False:
        return 10
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df, frequency=52)
    shrunk_cov = cs.shrunk_covariance(0)
    S = risk_models.sample_cov(df, frequency=52)
    np.testing.assert_array_almost_equal(shrunk_cov.values, S)

def test_ledoit_wolf_default():
    if False:
        print('Hello World!')
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    shrunk_cov = cs.ledoit_wolf()
    assert 0 < cs.delta < 1
    assert shrunk_cov.shape == (20, 20)
    assert list(shrunk_cov.index) == list(df.columns)
    assert list(shrunk_cov.columns) == list(df.columns)
    assert not shrunk_cov.isnull().any().any()
    assert risk_models._is_positive_semidefinite(shrunk_cov)

def test_ledoit_wolf_single_index():
    if False:
        for i in range(10):
            print('nop')
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    shrunk_cov = cs.ledoit_wolf(shrinkage_target='single_factor')
    assert 0 < cs.delta < 1
    assert shrunk_cov.shape == (20, 20)
    assert list(shrunk_cov.index) == list(df.columns)
    assert list(shrunk_cov.columns) == list(df.columns)
    assert not shrunk_cov.isnull().any().any()
    assert risk_models._is_positive_semidefinite(shrunk_cov)

def test_ledoit_wolf_constant_correlation():
    if False:
        for i in range(10):
            print('nop')
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    shrunk_cov = cs.ledoit_wolf(shrinkage_target='constant_correlation')
    assert 0 < cs.delta < 1
    assert shrunk_cov.shape == (20, 20)
    assert list(shrunk_cov.index) == list(df.columns)
    assert list(shrunk_cov.columns) == list(df.columns)
    assert not shrunk_cov.isnull().any().any()
    assert risk_models._is_positive_semidefinite(shrunk_cov)

def test_ledoit_wolf_raises_not_implemented():
    if False:
        for i in range(10):
            print('nop')
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    with pytest.raises(NotImplementedError):
        cs.ledoit_wolf(shrinkage_target='I have not been implemented!')

def test_oracle_approximating():
    if False:
        print('Hello World!')
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    shrunk_cov = cs.oracle_approximating()
    assert 0 < cs.delta < 1
    assert shrunk_cov.shape == (20, 20)
    assert list(shrunk_cov.index) == list(df.columns)
    assert list(shrunk_cov.columns) == list(df.columns)
    assert not shrunk_cov.isnull().any().any()
    assert risk_models._is_positive_semidefinite(shrunk_cov)

def test_risk_matrix_and_returns_data():
    if False:
        for i in range(10):
            print('nop')
    df = get_data()
    for method in {'sample_cov', 'semicovariance', 'exp_cov', 'ledoit_wolf', 'ledoit_wolf_constant_variance', 'ledoit_wolf_single_factor', 'ledoit_wolf_constant_correlation', 'oracle_approximating'}:
        S = risk_models.risk_matrix(df, method=method)
        assert S.shape == (20, 20)
        assert S.notnull().all().all()
        assert risk_models._is_positive_semidefinite(S)
        S2 = risk_models.risk_matrix(expected_returns.returns_from_prices(df), returns_data=True, method=method)
        pd.testing.assert_frame_equal(S, S2)

def test_risk_matrix_additional_kwargs():
    if False:
        for i in range(10):
            print('nop')
    df = get_data()
    S = risk_models.sample_cov(df)
    S2 = risk_models.risk_matrix(df, frequency=2)
    pd.testing.assert_frame_equal(S / 126, S2)
    S = risk_models.risk_matrix(df, method='semicovariance', benchmark=0.0004, frequency=52)
    assert S.shape == (20, 20)
    assert S.notnull().all().all()
    assert risk_models._is_positive_semidefinite(S)
    S = risk_models.risk_matrix(expected_returns.returns_from_prices(df), returns_data=True, method='exp_cov', span=60, fix_method='diag')
    assert S.shape == (20, 20)
    assert S.notnull().all().all()
    assert risk_models._is_positive_semidefinite(S)

def test_risk_matrix_not_implemented():
    if False:
        while True:
            i = 10
    df = get_data()
    with pytest.raises(NotImplementedError):
        risk_models.risk_matrix(df, method='fancy_new!')