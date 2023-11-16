from __future__ import absolute_import, division, print_function, unicode_literals
import backtrader as bt
import backtrader.indicators as btind

class RelativeVolume(bt.Indicator):
    csv = True
    lines = ('relvol',)
    params = (('period', 20), ('volisnan', True))

    def __init__(self):
        if False:
            while True:
                i = 10
        if self.p.volisnan:
            relvol = self.data.volume(-self.p.period) / self.data.volume
        else:
            relvol = bt.DivByZero(self.data.volume(-self.p.period), self.data.volume, zero=0.0)
        self.lines.relvol = relvol