import math
import numpy as np
import pytest
import pandas as pd
from pandas import Series, isna
import pandas._testing as tm

class TestSeriesCov:

    def test_cov(self, datetime_series):
        if False:
            i = 10
            return i + 15
        tm.assert_almost_equal(datetime_series.cov(datetime_series), datetime_series.std() ** 2)
        tm.assert_almost_equal(datetime_series[:15].cov(datetime_series[5:]), datetime_series[5:15].std() ** 2)
        assert np.isnan(datetime_series[::2].cov(datetime_series[1::2]))
        cp = datetime_series[:10].copy()
        cp[:] = np.nan
        assert isna(cp.cov(cp))
        assert isna(datetime_series[:15].cov(datetime_series[5:], min_periods=12))
        ts1 = datetime_series[:15].reindex(datetime_series.index)
        ts2 = datetime_series[5:].reindex(datetime_series.index)
        assert isna(ts1.cov(ts2, min_periods=12))

    @pytest.mark.parametrize('test_ddof', [None, 0, 1, 2, 3])
    @pytest.mark.parametrize('dtype', ['float64', 'Float64'])
    def test_cov_ddof(self, test_ddof, dtype):
        if False:
            return 10
        np_array1 = np.random.default_rng(2).random(10)
        np_array2 = np.random.default_rng(2).random(10)
        s1 = Series(np_array1, dtype=dtype)
        s2 = Series(np_array2, dtype=dtype)
        result = s1.cov(s2, ddof=test_ddof)
        expected = np.cov(np_array1, np_array2, ddof=test_ddof)[0][1]
        assert math.isclose(expected, result)

class TestSeriesCorr:

    @pytest.mark.parametrize('dtype', ['float64', 'Float64'])
    def test_corr(self, datetime_series, dtype):
        if False:
            for i in range(10):
                print('nop')
        stats = pytest.importorskip('scipy.stats')
        datetime_series = datetime_series.astype(dtype)
        tm.assert_almost_equal(datetime_series.corr(datetime_series), 1)
        tm.assert_almost_equal(datetime_series[:15].corr(datetime_series[5:]), 1)
        assert isna(datetime_series[:15].corr(datetime_series[5:], min_periods=12))
        ts1 = datetime_series[:15].reindex(datetime_series.index)
        ts2 = datetime_series[5:].reindex(datetime_series.index)
        assert isna(ts1.corr(ts2, min_periods=12))
        assert np.isnan(datetime_series[::2].corr(datetime_series[1::2]))
        cp = datetime_series[:10].copy()
        cp[:] = np.nan
        assert isna(cp.corr(cp))
        A = tm.makeTimeSeries()
        B = tm.makeTimeSeries()
        result = A.corr(B)
        (expected, _) = stats.pearsonr(A, B)
        tm.assert_almost_equal(result, expected)

    def test_corr_rank(self):
        if False:
            for i in range(10):
                print('nop')
        stats = pytest.importorskip('scipy.stats')
        A = tm.makeTimeSeries()
        B = tm.makeTimeSeries()
        A[-5:] = A[:5].copy()
        result = A.corr(B, method='kendall')
        expected = stats.kendalltau(A, B)[0]
        tm.assert_almost_equal(result, expected)
        result = A.corr(B, method='spearman')
        expected = stats.spearmanr(A, B)[0]
        tm.assert_almost_equal(result, expected)
        A = Series([-0.89926396, 0.94209606, -1.03289164, -0.95445587, 0.7691031, -0.06430576, -2.09704447, 0.40660407, -0.89926396, 0.94209606])
        B = Series([-1.01270225, -0.62210117, -1.56895827, 0.59592943, -0.01680292, 1.17258718, -1.06009347, -0.1022206, -0.89076239, 0.89372375])
        kexp = 0.4319297
        sexp = 0.5853767
        tm.assert_almost_equal(A.corr(B, method='kendall'), kexp)
        tm.assert_almost_equal(A.corr(B, method='spearman'), sexp)

    def test_corr_invalid_method(self):
        if False:
            i = 10
            return i + 15
        s1 = Series(np.random.default_rng(2).standard_normal(10))
        s2 = Series(np.random.default_rng(2).standard_normal(10))
        msg = "method must be either 'pearson', 'spearman', 'kendall', or a callable, "
        with pytest.raises(ValueError, match=msg):
            s1.corr(s2, method='____')

    def test_corr_callable_method(self, datetime_series):
        if False:
            while True:
                i = 10
        my_corr = lambda a, b: 1.0 if (a == b).all() else 0.0
        s1 = Series([1, 2, 3, 4, 5])
        s2 = Series([5, 4, 3, 2, 1])
        expected = 0
        tm.assert_almost_equal(s1.corr(s2, method=my_corr), expected)
        tm.assert_almost_equal(datetime_series.corr(datetime_series, method=my_corr), 1.0)
        tm.assert_almost_equal(datetime_series[:15].corr(datetime_series[5:], method=my_corr), 1.0)
        assert np.isnan(datetime_series[::2].corr(datetime_series[1::2], method=my_corr))
        df = pd.DataFrame([s1, s2])
        expected = pd.DataFrame([{0: 1.0, 1: 0}, {0: 0, 1: 1.0}])
        tm.assert_almost_equal(df.transpose().corr(method=my_corr), expected)