import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.stats._tukeylambda_stats import tukeylambda_variance, tukeylambda_kurtosis

def test_tukeylambda_stats_known_exact():
    if False:
        while True:
            i = 10
    'Compare results with some known exact formulas.'
    var = tukeylambda_variance(0)
    assert_allclose(var, np.pi ** 2 / 3, atol=1e-12)
    kurt = tukeylambda_kurtosis(0)
    assert_allclose(kurt, 1.2, atol=1e-10)
    var = tukeylambda_variance(0.5)
    assert_allclose(var, 4 - np.pi, atol=1e-12)
    kurt = tukeylambda_kurtosis(0.5)
    desired = (5.0 / 3 - np.pi / 2) / (np.pi / 4 - 1) ** 2 - 3
    assert_allclose(kurt, desired, atol=1e-10)
    var = tukeylambda_variance(1)
    assert_allclose(var, 1.0 / 3, atol=1e-12)
    kurt = tukeylambda_kurtosis(1)
    assert_allclose(kurt, -1.2, atol=1e-10)
    var = tukeylambda_variance(2)
    assert_allclose(var, 1.0 / 12, atol=1e-12)
    kurt = tukeylambda_kurtosis(2)
    assert_allclose(kurt, -1.2, atol=1e-10)

def test_tukeylambda_stats_mpmath():
    if False:
        print('Hello World!')
    'Compare results with some values that were computed using mpmath.'
    a10 = dict(atol=1e-10, rtol=0)
    a12 = dict(atol=1e-12, rtol=0)
    data = [[-0.1, 4.780502178742536, 3.785595203464545], [-0.0649, 4.164280235998958, 2.520196759474357], [-0.05, 3.9367226789077527, 2.1312979305777726], [-0.001, 3.301283803909649, 1.2145246008354298], [0.001, 3.278507756495722, 1.1856063477928758], [0.03125, 2.959278032546158, 0.80448755516182], [0.05, 2.782810534054645, 0.6116040438866444], [0.0649, 2.6528238675410054, 0.47683411953277455], [1.2, 0.24215392057858834, -1.2342804716904971], [10.0, 0.000952375797577036, 2.3781069735514495], [20.0, 0.00012195121951131043, 7.376543210027095]]
    for (lam, var_expected, kurt_expected) in data:
        var = tukeylambda_variance(lam)
        assert_allclose(var, var_expected, **a12)
        kurt = tukeylambda_kurtosis(lam)
        assert_allclose(kurt, kurt_expected, **a10)
    (lam, var_expected, kurt_expected) = zip(*data)
    var = tukeylambda_variance(lam)
    assert_allclose(var, var_expected, **a12)
    kurt = tukeylambda_kurtosis(lam)
    assert_allclose(kurt, kurt_expected, **a10)

def test_tukeylambda_stats_invalid():
    if False:
        while True:
            i = 10
    'Test values of lambda outside the domains of the functions.'
    lam = [-1.0, -0.5]
    var = tukeylambda_variance(lam)
    assert_equal(var, np.array([np.nan, np.inf]))
    lam = [-1.0, -0.25]
    kurt = tukeylambda_kurtosis(lam)
    assert_equal(kurt, np.array([np.nan, np.inf]))