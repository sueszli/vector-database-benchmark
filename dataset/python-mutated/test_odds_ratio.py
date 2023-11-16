import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from .._discrete_distns import nchypergeom_fisher, hypergeom
from scipy.stats._odds_ratio import odds_ratio
from .data.fisher_exact_results_from_r import data

class TestOddsRatio:

    @pytest.mark.parametrize('parameters, rresult', data)
    def test_results_from_r(self, parameters, rresult):
        if False:
            i = 10
            return i + 15
        alternative = parameters.alternative.replace('.', '-')
        result = odds_ratio(parameters.table)
        if result.statistic < 400:
            or_rtol = 0.0005
            ci_rtol = 0.02
        else:
            or_rtol = 0.05
            ci_rtol = 0.1
        assert_allclose(result.statistic, rresult.conditional_odds_ratio, rtol=or_rtol)
        ci = result.confidence_interval(parameters.confidence_level, alternative)
        assert_allclose((ci.low, ci.high), rresult.conditional_odds_ratio_ci, rtol=ci_rtol)
        cor = result.statistic
        table = np.array(parameters.table)
        total = table.sum()
        ngood = table[0].sum()
        nsample = table[:, 0].sum()
        if cor == 0:
            nchg_mean = hypergeom.support(total, ngood, nsample)[0]
        elif cor == np.inf:
            nchg_mean = hypergeom.support(total, ngood, nsample)[1]
        else:
            nchg_mean = nchypergeom_fisher.mean(total, ngood, nsample, cor)
        assert_allclose(nchg_mean, table[0, 0], rtol=1e-13)
        alpha = 1 - parameters.confidence_level
        if alternative == 'two-sided':
            if ci.low > 0:
                sf = nchypergeom_fisher.sf(table[0, 0] - 1, total, ngood, nsample, ci.low)
                assert_allclose(sf, alpha / 2, rtol=1e-11)
            if np.isfinite(ci.high):
                cdf = nchypergeom_fisher.cdf(table[0, 0], total, ngood, nsample, ci.high)
                assert_allclose(cdf, alpha / 2, rtol=1e-11)
        elif alternative == 'less':
            if np.isfinite(ci.high):
                cdf = nchypergeom_fisher.cdf(table[0, 0], total, ngood, nsample, ci.high)
                assert_allclose(cdf, alpha, rtol=1e-11)
        elif ci.low > 0:
            sf = nchypergeom_fisher.sf(table[0, 0] - 1, total, ngood, nsample, ci.low)
            assert_allclose(sf, alpha, rtol=1e-11)

    @pytest.mark.parametrize('table', [[[0, 0], [5, 10]], [[5, 10], [0, 0]], [[0, 5], [0, 10]], [[5, 0], [10, 0]]])
    def test_row_or_col_zero(self, table):
        if False:
            for i in range(10):
                print('nop')
        result = odds_ratio(table)
        assert_equal(result.statistic, np.nan)
        ci = result.confidence_interval()
        assert_equal((ci.low, ci.high), (0, np.inf))

    @pytest.mark.parametrize('case', [[0.95, 'two-sided', 0.4879913, 2.635883], [0.9, 'two-sided', 0.5588516, 2.301663]])
    def test_sample_odds_ratio_ci(self, case):
        if False:
            for i in range(10):
                print('nop')
        (confidence_level, alternative, ref_low, ref_high) = case
        table = [[10, 20], [41, 93]]
        result = odds_ratio(table, kind='sample')
        assert_allclose(result.statistic, 1.134146, rtol=1e-06)
        ci = result.confidence_interval(confidence_level, alternative)
        assert_allclose([ci.low, ci.high], [ref_low, ref_high], rtol=1e-06)

    @pytest.mark.parametrize('alternative', ['less', 'greater', 'two-sided'])
    def test_sample_odds_ratio_one_sided_ci(self, alternative):
        if False:
            i = 10
            return i + 15
        table = [[1000, 2000], [4100, 9300]]
        res = odds_ratio(table, kind='sample')
        ref = odds_ratio(table, kind='conditional')
        assert_allclose(res.statistic, ref.statistic, atol=1e-05)
        assert_allclose(res.confidence_interval(alternative=alternative), ref.confidence_interval(alternative=alternative), atol=0.002)

    @pytest.mark.parametrize('kind', ['sample', 'conditional'])
    @pytest.mark.parametrize('bad_table', [123, 'foo', [10, 11, 12]])
    def test_invalid_table_shape(self, kind, bad_table):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError, match='Invalid shape'):
            odds_ratio(bad_table, kind=kind)

    def test_invalid_table_type(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError, match='must be an array of integers'):
            odds_ratio([[1.0, 3.4], [5.0, 9.9]])

    def test_negative_table_values(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError, match='must be nonnegative'):
            odds_ratio([[1, 2], [3, -4]])

    def test_invalid_kind(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError, match='`kind` must be'):
            odds_ratio([[10, 20], [30, 14]], kind='magnetoreluctance')

    def test_invalid_alternative(self):
        if False:
            return 10
        result = odds_ratio([[5, 10], [2, 32]])
        with pytest.raises(ValueError, match='`alternative` must be'):
            result.confidence_interval(alternative='depleneration')

    @pytest.mark.parametrize('level', [-0.5, 1.5])
    def test_invalid_confidence_level(self, level):
        if False:
            return 10
        result = odds_ratio([[5, 10], [2, 32]])
        with pytest.raises(ValueError, match='must be between 0 and 1'):
            result.confidence_interval(confidence_level=level)