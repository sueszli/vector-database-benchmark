from __future__ import absolute_import, division, print_function, unicode_literals
import backtrader as bt
from . import PeriodN
__all__ = ['OLS_Slope_InterceptN', 'OLS_TransformationN', 'OLS_BetaN', 'CointN']

class OLS_Slope_InterceptN(PeriodN):
    """
    Calculates a linear regression using ``statsmodel.OLS`` (Ordinary least
    squares) of data1 on data0

    Uses ``pandas`` and ``statsmodels``
    """
    _mindatas = 2
    packages = (('pandas', 'pd'), ('statsmodels.api', 'sm'))
    lines = ('slope', 'intercept')
    params = (('period', 10),)

    def next(self):
        if False:
            while True:
                i = 10
        p0 = pd.Series(self.data0.get(size=self.p.period))
        p1 = pd.Series(self.data1.get(size=self.p.period))
        p1 = sm.add_constant(p1)
        (intercept, slope) = sm.OLS(p0, p1).fit().params
        self.lines.slope[0] = slope
        self.lines.intercept[0] = intercept

class OLS_TransformationN(PeriodN):
    """
    Calculates the ``zscore`` for data0 and data1. Although it doesn't directly
    uses any external package it relies on ``OLS_SlopeInterceptN`` which uses
    ``pandas`` and ``statsmodels``
    """
    _mindatas = 2
    lines = ('spread', 'spread_mean', 'spread_std', 'zscore')
    params = (('period', 10),)

    def __init__(self):
        if False:
            while True:
                i = 10
        slint = OLS_Slope_InterceptN(*self.datas)
        spread = self.data0 - (slint.slope * self.data1 + slint.intercept)
        self.l.spread = spread
        self.l.spread_mean = bt.ind.SMA(spread, period=self.p.period)
        self.l.spread_std = bt.ind.StdDev(spread, period=self.p.period)
        self.l.zscore = (spread - self.l.spread_mean) / self.l.spread_std

class OLS_BetaN(PeriodN):
    """
    Calculates a regression of data1 on data0 using ``pandas.ols``

    Uses ``pandas``
    """
    _mindatas = 2
    packages = (('pandas', 'pd'),)
    lines = ('beta',)
    params = (('period', 10),)

    def next(self):
        if False:
            return 10
        (y, x) = (pd.Series(d.get(size=self.p.period)) for d in self.datas)
        r_beta = pd.ols(y=y, x=x, window_type='full_sample')
        self.lines.beta[0] = r_beta.beta['x']

class CointN(PeriodN):
    """
    Calculates the score (coint_t) and pvalue for a given ``period`` for the
    data feeds

    Uses ``pandas`` and ``statsmodels`` (for ``coint``)
    """
    _mindatas = 2
    packages = (('pandas', 'pd'),)
    frompackages = (('statsmodels.tsa.stattools', 'coint'),)
    lines = ('score', 'pvalue')
    params = (('period', 10), ('trend', 'c'))

    def next(self):
        if False:
            return 10
        (x, y) = (pd.Series(d.get(size=self.p.period)) for d in self.datas)
        (score, pvalue, _) = coint(x, y, trend=self.p.trend)
        self.lines.score[0] = score
        self.lines.pvalue[0] = pvalue