import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.special import stdtr, stdtrit, ndtr, ndtri

def test_stdtr_vs_R_large_df():
    if False:
        for i in range(10):
            print('nop')
    df = [10000000000.0, 1000000000000.0, 1e+120, np.inf]
    t = 1.0
    res = stdtr(df, t)
    res_R = [0.8413447460564446, 0.8413447460684218, 0.8413447460685428, 0.8413447460685429]
    assert_allclose(res, res_R, rtol=2e-15)
    assert_equal(res[3], ndtr(1.0))

def test_stdtrit_vs_R_large_df():
    if False:
        while True:
            i = 10
    df = [10000000000.0, 1000000000000.0, 1e+120, np.inf]
    p = 0.1
    res = stdtrit(df, p)
    res_R = [-1.2815515656292593, -1.2815515655454472, -1.2815515655446008, -1.2815515655446008]
    assert_allclose(res, res_R, rtol=1e-15)
    assert_equal(res[3], ndtri(0.1))

def test_stdtr_stdtri_invalid():
    if False:
        while True:
            i = 10
    df = [10000000000.0, 1000000000000.0, 1e+120, np.inf]
    x = np.nan
    res1 = stdtr(df, x)
    res2 = stdtrit(df, x)
    res_ex = 4 * [np.nan]
    assert_equal(res1, res_ex)
    assert_equal(res2, res_ex)