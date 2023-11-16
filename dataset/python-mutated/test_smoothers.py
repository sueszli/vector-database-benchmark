"""
Created on Fri Nov 04 10:51:39 2011

Author: Josef Perktold
License: BSD-3
"""
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.sandbox.nonparametric import smoothers
from statsmodels.regression.linear_model import OLS, WLS

class CheckSmoother:

    def test_predict(self):
        if False:
            print('Hello World!')
        assert_almost_equal(self.res_ps.predict(self.x), self.res2.fittedvalues, decimal=13)
        assert_almost_equal(self.res_ps.predict(self.x[:10]), self.res2.fittedvalues[:10], decimal=13)

    def test_coef(self):
        if False:
            i = 10
            return i + 15
        assert_almost_equal(self.res_ps.coef.ravel(), self.res2.params, decimal=14)

    def test_df(self):
        if False:
            i = 10
            return i + 15
        assert_equal(self.res_ps.df_model(), self.res2.df_model + 1)
        assert_equal(self.res_ps.df_fit(), self.res2.df_model + 1)
        assert_equal(self.res_ps.df_resid(), self.res2.df_resid)

class BasePolySmoother:

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        order = 3
        sigma_noise = 0.5
        nobs = 100
        (lb, ub) = (-1, 2)
        cls.x = x = np.linspace(lb, ub, nobs)
        cls.exog = exog = x[:, None] ** np.arange(order + 1)
        y_true = exog.sum(1)
        np.random.seed(987567)
        cls.y = y = y_true + sigma_noise * np.random.randn(nobs)

class TestPolySmoother1(BasePolySmoother, CheckSmoother):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        super(TestPolySmoother1, cls).setup_class()
        (y, x, exog) = (cls.y, cls.x, cls.exog)
        pmod = smoothers.PolySmoother(2, x)
        pmod.fit(y)
        cls.res_ps = pmod
        cls.res2 = OLS(y, exog[:, :2 + 1]).fit()

class TestPolySmoother2(BasePolySmoother, CheckSmoother):

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        super(TestPolySmoother2, cls).setup_class()
        (y, x, exog) = (cls.y, cls.x, cls.exog)
        pmod = smoothers.PolySmoother(3, x)
        pmod.smooth(y)
        cls.res_ps = pmod
        cls.res2 = OLS(y, exog[:, :3 + 1]).fit()

class TestPolySmoother3(BasePolySmoother, CheckSmoother):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        super(TestPolySmoother3, cls).setup_class()
        (y, x, exog) = (cls.y, cls.x, cls.exog)
        nobs = y.shape[0]
        weights = np.ones(nobs)
        weights[:nobs // 3] = 0.1
        weights[-nobs // 5:] = 2
        pmod = smoothers.PolySmoother(2, x)
        pmod.fit(y, weights=weights)
        cls.res_ps = pmod
        cls.res2 = WLS(y, exog[:, :2 + 1], weights=weights).fit()
if __name__ == '__main__':
    t1 = TestPolySmoother1()
    t1.test_predict()
    t1.test_coef()
    t1.test_df
    t3 = TestPolySmoother3()
    t3.test_predict()