from __future__ import absolute_import, division, print_function, unicode_literals
import backtrader as bt
from . import MovAv
__all__ = ['haDelta', 'haD']

class haDelta(bt.Indicator):
    """Heikin Ashi Delta. Defined by Dan Valcu in his book "Heikin-Ashi: How to
    Trade Without Candlestick Patterns ".

    This indicator measures difference between Heikin Ashi close and open of
    Heikin Ashi candles, the body of the candle.

    To get signals add haDelta smoothed by 3 period moving average.

    For correct use, the data for the indicator must have been previously
    passed by the Heikin Ahsi filter.

    Formula:
      - haDelta = Heikin Ashi close - Heikin Ashi open
      - smoothed = movav(haDelta, period)

    """
    alias = ('haD',)
    lines = ('haDelta', 'smoothed')
    params = (('period', 3), ('movav', MovAv.SMA), ('autoheikin', True))
    plotinfo = dict(subplot=True)
    plotlines = dict(haDelta=dict(color='red'), smoothed=dict(color='grey', _fill_gt=(0, 'green'), _fill_lt=(0, 'red')))

    def __init__(self):
        if False:
            i = 10
            return i + 15
        d = bt.ind.HeikinAshi(self.data) if self.p.autoheikin else self.data
        self.lines.haDelta = hd = d.close - d.open
        self.lines.smoothed = self.p.movav(hd, period=self.p.period)
        super(haDelta, self).__init__()