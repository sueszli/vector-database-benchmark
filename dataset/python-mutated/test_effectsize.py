"""
Created on Mon Oct  5 13:13:59 2020

Author: Josef Perktold
License: BSD-3

"""
from scipy import stats
from numpy.testing import assert_allclose
from statsmodels.stats.effect_size import _noncentrality_chisquare, _noncentrality_f, _noncentrality_t

def test_noncent_chi2():
    if False:
        for i in range(10):
            print('nop')
    (chi2_stat, df) = (7.5, 2)
    ci_nc = [0.03349255, 20.76049805]
    res = _noncentrality_chisquare(chi2_stat, df, alpha=0.05)
    assert_allclose(res.confint, ci_nc, rtol=0.005)
    mean = stats.ncx2.mean(df, res.nc)
    assert_allclose(chi2_stat, mean, rtol=1e-08)
    assert_allclose(stats.ncx2.cdf(chi2_stat, df, res.confint), [0.975, 0.025], rtol=1e-08)

def test_noncent_f():
    if False:
        i = 10
        return i + 15
    (f_stat, df1, df2) = (3.5, 4, 75)
    ci_nc = [0.7781436, 29.72949219]
    res = _noncentrality_f(f_stat, df1, df2, alpha=0.05)
    assert_allclose(res.confint, ci_nc, rtol=0.005)
    mean = stats.ncf.mean(df1, df2, res.nc)
    assert_allclose(f_stat, mean, rtol=1e-08)
    assert_allclose(stats.ncf.cdf(f_stat, df1, df2, res.confint), [0.975, 0.025], rtol=5e-05)

def test_noncent_t():
    if False:
        for i in range(10):
            print('nop')
    (t_stat, df) = (1.5, 98)
    ci_nc = [-0.474934, 3.467371]
    res = _noncentrality_t(t_stat, df, alpha=0.05)
    assert_allclose(res.confint, ci_nc, rtol=0.005)
    mean = stats.nct.mean(df, res.nc)
    assert_allclose(t_stat, mean, rtol=1e-08)
    assert_allclose(stats.nct.cdf(t_stat, df, res.confint), [0.975, 0.025], rtol=1e-06)