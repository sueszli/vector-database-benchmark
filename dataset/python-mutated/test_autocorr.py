import numpy as np

class TestAutoCorr:

    def test_autocorr(self, datetime_series):
        if False:
            return 10
        corr1 = datetime_series.autocorr()
        corr2 = datetime_series.autocorr(lag=1)
        if len(datetime_series) <= 2:
            assert np.isnan(corr1)
            assert np.isnan(corr2)
        else:
            assert corr1 == corr2
        n = 1 + np.random.default_rng(2).integers(max(1, len(datetime_series) - 2))
        corr1 = datetime_series.corr(datetime_series.shift(n))
        corr2 = datetime_series.autocorr(lag=n)
        if len(datetime_series) <= 2:
            assert np.isnan(corr1)
            assert np.isnan(corr2)
        else:
            assert corr1 == corr2