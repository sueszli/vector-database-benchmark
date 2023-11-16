import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal, assert_raises, assert_equal, assert_allclose
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import omni_normtest, jarque_bera, durbin_watson, _medcouple_1d, medcouple, robust_kurtosis, robust_skewness
x = np.array([-0.1184, -1.3403, 0.0063, -0.612, -0.3869, -0.2313, -2.8485, -0.2167, 0.4153, 1.8492, -0.3706, 0.9726, -0.1501, -0.0337, -1.4423, 1.2489, 0.9182, -0.2331, -0.6182, 0.183])

def test_durbin_watson():
    if False:
        for i in range(10):
            print('nop')
    st_R = 1.95298958377419
    assert_almost_equal(durbin_watson(x), st_R, 14)
    st_R = 1.848802400319998
    assert_almost_equal(durbin_watson(x ** 2), st_R, 14)
    st_R = 1.09897993228779
    assert_almost_equal(durbin_watson(x[1:] + 0.5 * x[:-1]), st_R, 14)
    st_R = 0.937241876707273
    assert_almost_equal(durbin_watson(x[1:] + 0.8 * x[:-1]), st_R, 14)
    st_R = 0.921488912587806
    assert_almost_equal(durbin_watson(x[1:] + 0.9 * x[:-1]), st_R, 14)
    X = np.array([x, x])
    st_R = 1.95298958377419
    assert_almost_equal(durbin_watson(X, axis=1), np.array([st_R, st_R]), 14)
    assert_almost_equal(durbin_watson(X.T, axis=0), np.array([st_R, st_R]), 14)

def test_omni_normtest():
    if False:
        return 10
    from scipy import stats
    st_pv_R = np.array([[3.994138321207883, -1.12930430216146, 1.648881473704978], [0.1357325110375005, 0.2587694866795507, 0.0991719192710234]])
    nt = omni_normtest(x)
    assert_almost_equal(nt, st_pv_R[:, 0], 14)
    st = stats.skewtest(x)
    assert_almost_equal(st, st_pv_R[:, 1], 14)
    kt = stats.kurtosistest(x)
    assert_almost_equal(kt, st_pv_R[:, 2], 11)
    st_pv_R = np.array([[34.523210399523926, 4.429509162503833, 3.860396220444025], [3.186985686465249e-08, 9.444780064482572e-06, 0.0001132033129378485]])
    x2 = x ** 2
    nt = omni_normtest(x2)
    assert_almost_equal(nt, st_pv_R[:, 0], 12)
    st = stats.skewtest(x2)
    assert_almost_equal(st, st_pv_R[:, 1], 12)
    kt = stats.kurtosistest(x2)
    assert_almost_equal(kt, st_pv_R[:, 2], 12)

def test_omni_normtest_axis(reset_randomstate):
    if False:
        for i in range(10):
            print('nop')
    x = np.random.randn(25, 3)
    nt1 = omni_normtest(x)
    nt2 = omni_normtest(x, axis=0)
    nt3 = omni_normtest(x.T, axis=1)
    assert_almost_equal(nt2, nt1, decimal=13)
    assert_almost_equal(nt3, nt1, decimal=13)

def test_jarque_bera():
    if False:
        print('Hello World!')
    st_pv_R = np.array([1.966267722686169, 0.3741367669648314])
    jb = jarque_bera(x)[:2]
    assert_almost_equal(jb, st_pv_R, 14)
    st_pv_R = np.array([78.329987305556, 0.0])
    jb = jarque_bera(x ** 2)[:2]
    assert_almost_equal(jb, st_pv_R, 13)
    st_pv_R = np.array([5.713575079670667, 0.0574530296971343])
    jb = jarque_bera(np.log(x ** 2))[:2]
    assert_almost_equal(jb, st_pv_R, 14)
    st_pv_R = np.array([2.648931574849576, 0.2659449923067881])
    jb = jarque_bera(np.exp(-x ** 2))[:2]
    assert_almost_equal(jb, st_pv_R, 14)

def test_shapiro():
    if False:
        return 10
    from scipy.stats import shapiro
    st_pv_R = np.array([0.939984787255526, 0.23962189800046])
    sh = shapiro(x)
    assert_almost_equal(sh, st_pv_R, 4)
    st_pv_R = np.array([0.5799574255943298, 1.838456834681376e-06 * 10000.0])
    sh = shapiro(x ** 2) * np.array([1, 10000.0])
    assert_almost_equal(sh, st_pv_R, 5)
    st_pv_R = np.array([0.9173044264316559, 0.08793704167882448])
    sh = shapiro(np.log(x ** 2))
    assert_almost_equal(sh, st_pv_R, 5)
    st_pv_R = np.array([0.8183618634939194, 0.001644620895206969])
    sh = shapiro(np.exp(-x ** 2))
    assert_almost_equal(sh, st_pv_R, 5)

def test_adnorm():
    if False:
        i = 10
        return i + 15
    st_pv = []
    st_pv_R = np.array([0.5867235358882148, 0.1115380760041617])
    ad = normal_ad(x)
    assert_almost_equal(ad, st_pv_R, 12)
    st_pv.append(st_pv_R)
    st_pv_R = np.array([2.976266267594575, 8.753003709960645e-08])
    ad = normal_ad(x ** 2)
    assert_almost_equal(ad, st_pv_R, 11)
    st_pv.append(st_pv_R)
    st_pv_R = np.array([0.4892557856308528, 0.1968040759316307])
    ad = normal_ad(np.log(x ** 2))
    assert_almost_equal(ad, st_pv_R, 12)
    st_pv.append(st_pv_R)
    st_pv_R = np.array([1.459901465428267, 0.0006380009232897535])
    ad = normal_ad(np.exp(-x ** 2))
    assert_almost_equal(ad, st_pv_R, 12)
    st_pv.append(st_pv_R)
    ad = normal_ad(np.column_stack((x, x ** 2, np.log(x ** 2), np.exp(-x ** 2))).T, axis=1)
    assert_almost_equal(ad, np.column_stack(st_pv), 11)

def test_durbin_watson_pandas(reset_randomstate):
    if False:
        for i in range(10):
            print('nop')
    x = np.random.randn(50)
    x_series = pd.Series(x)
    assert_almost_equal(durbin_watson(x), durbin_watson(x_series), decimal=13)

class TestStattools:

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        x = np.random.standard_normal(1000)
        (e1, e2, e3, e4, e5, e6, e7) = np.percentile(x, (12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5))
        (c05, c50, c95) = np.percentile(x, (5.0, 50.0, 95.0))
        (f025, f25, f75, f975) = np.percentile(x, (2.5, 25.0, 75.0, 97.5))
        mean = np.mean
        kr1 = mean(((x - mean(x)) / np.std(x)) ** 4.0) - 3.0
        kr2 = (e7 - e5 + (e3 - e1)) / (e6 - e2) - 1.2330951154852172
        kr3 = (mean(x[x > c95]) - mean(x[x < c05])) / (mean(x[x > c50]) - mean(x[x < c50])) - 2.585227122870805
        kr4 = (f975 - f025) / (f75 - f25) - 2.905846951670164
        cls.kurtosis_x = x
        cls.expected_kurtosis = np.array([kr1, kr2, kr3, kr4])
        cls.kurtosis_constants = np.array([3.0, 1.2330951154852172, 2.585227122870805, 2.905846951670164])

    def test_medcouple_no_axis(self):
        if False:
            i = 10
            return i + 15
        x = np.reshape(np.arange(100.0), (50, 2))
        mc = medcouple(x, axis=None)
        assert_almost_equal(mc, medcouple(x.ravel()))

    def test_medcouple_1d(self):
        if False:
            i = 10
            return i + 15
        x = np.reshape(np.arange(100.0), (50, 2))
        assert_raises(ValueError, _medcouple_1d, x)

    def test_medcouple_symmetric(self):
        if False:
            return 10
        mc = medcouple(np.arange(5.0))
        assert_almost_equal(mc, 0)

    def test_medcouple_nonzero(self):
        if False:
            print('Hello World!')
        mc = medcouple(np.array([1, 2, 7, 9, 10.0]))
        assert_almost_equal(mc, -0.3333333)

    def test_medcouple_int(self):
        if False:
            print('Hello World!')
        mc1 = medcouple(np.array([1, 2, 7, 9, 10]))
        mc2 = medcouple(np.array([1, 2, 7, 9, 10.0]))
        assert_equal(mc1, mc2)

    def test_medcouple_symmetry(self, reset_randomstate):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.standard_normal(100)
        mcp = medcouple(x)
        mcn = medcouple(-x)
        assert_almost_equal(mcp + mcn, 0)

    def test_medcouple_ties(self, reset_randomstate):
        if False:
            print('Hello World!')
        x = np.array([1, 2, 2, 3, 4])
        mc = medcouple(x)
        assert_almost_equal(mc, 1.0 / 6.0)

    def test_durbin_watson(self, reset_randomstate):
        if False:
            while True:
                i = 10
        x = np.random.standard_normal(100)
        dw = sum(np.diff(x) ** 2.0) / np.dot(x, x)
        assert_almost_equal(dw, durbin_watson(x))

    def test_durbin_watson_2d(self, reset_randomstate):
        if False:
            for i in range(10):
                print('nop')
        shape = (1, 10)
        x = np.random.standard_normal(100)
        dw = sum(np.diff(x) ** 2.0) / np.dot(x, x)
        x = np.tile(x[:, None], shape)
        assert_almost_equal(np.squeeze(dw * np.ones(shape)), durbin_watson(x))

    def test_durbin_watson_3d(self, reset_randomstate):
        if False:
            return 10
        shape = (10, 1, 10)
        x = np.random.standard_normal(100)
        dw = sum(np.diff(x) ** 2.0) / np.dot(x, x)
        x = np.tile(x[None, :, None], shape)
        assert_almost_equal(np.squeeze(dw * np.ones(shape)), durbin_watson(x, axis=1))

    def test_robust_skewness_1d(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(21.0)
        sk = robust_skewness(x)
        assert_almost_equal(np.array(sk), np.zeros(4))

    def test_robust_skewness_1d_2d(self, reset_randomstate):
        if False:
            print('Hello World!')
        x = np.random.randn(21)
        y = x[:, None]
        sk_x = robust_skewness(x)
        sk_y = robust_skewness(y, axis=None)
        assert_almost_equal(np.array(sk_x), np.array(sk_y))

    def test_robust_skewness_symmetric(self, reset_randomstate):
        if False:
            print('Hello World!')
        x = np.random.standard_normal(100)
        x = np.hstack([x, np.zeros(1), -x])
        sk = robust_skewness(x)
        assert_almost_equal(np.array(sk), np.zeros(4))

    def test_robust_skewness_3d(self, reset_randomstate):
        if False:
            i = 10
            return i + 15
        x = np.random.standard_normal(100)
        x = np.hstack([x, np.zeros(1), -x])
        x = np.tile(x, (10, 10, 1))
        sk_3d = robust_skewness(x, axis=2)
        result = np.zeros((10, 10))
        for sk in sk_3d:
            assert_almost_equal(sk, result)

    def test_robust_skewness_4(self, reset_randomstate):
        if False:
            i = 10
            return i + 15
        x = np.random.standard_normal(1000)
        x[x > 0] *= 3
        m = np.median(x)
        s = x.std(ddof=0)
        expected = (x.mean() - m) / s
        (_, _, _, sk4) = robust_skewness(x)
        assert_allclose(expected, sk4)

    def test_robust_kurtosis_1d_2d(self, reset_randomstate):
        if False:
            print('Hello World!')
        x = np.random.randn(100)
        y = x[:, None]
        kr_x = np.array(robust_kurtosis(x))
        kr_y = np.array(robust_kurtosis(y, axis=None))
        assert_almost_equal(kr_x, kr_y)

    def test_robust_kurtosis(self):
        if False:
            i = 10
            return i + 15
        x = self.kurtosis_x
        assert_almost_equal(np.array(robust_kurtosis(x)), self.expected_kurtosis)

    def test_robust_kurtosis_3d(self):
        if False:
            while True:
                i = 10
        x = np.tile(self.kurtosis_x, (10, 10, 1))
        kurtosis = np.array(robust_kurtosis(x, axis=2))
        for (i, r) in enumerate(self.expected_kurtosis):
            assert_almost_equal(r * np.ones((10, 10)), kurtosis[i])

    def test_robust_kurtosis_excess_false(self):
        if False:
            i = 10
            return i + 15
        x = self.kurtosis_x
        expected = self.expected_kurtosis + self.kurtosis_constants
        kurtosis = np.array(robust_kurtosis(x, excess=False))
        assert_almost_equal(expected, kurtosis)

    def test_robust_kurtosis_ab(self):
        if False:
            while True:
                i = 10
        x = self.kurtosis_x
        (alpha, beta) = (10.0, 45.0)
        kurtosis = robust_kurtosis(self.kurtosis_x, ab=(alpha, beta), excess=False)
        num = np.mean(x[x > np.percentile(x, 100.0 - alpha)]) - np.mean(x[x < np.percentile(x, alpha)])
        denom = np.mean(x[x > np.percentile(x, 100.0 - beta)]) - np.mean(x[x < np.percentile(x, beta)])
        assert_almost_equal(kurtosis[2], num / denom)

    def test_robust_kurtosis_dg(self):
        if False:
            print('Hello World!')
        x = self.kurtosis_x
        (delta, gamma) = (10.0, 45.0)
        kurtosis = robust_kurtosis(self.kurtosis_x, dg=(delta, gamma), excess=False)
        q = np.percentile(x, [delta, 100.0 - delta, gamma, 100.0 - gamma])
        assert_almost_equal(kurtosis[3], (q[1] - q[0]) / (q[3] - q[2]))