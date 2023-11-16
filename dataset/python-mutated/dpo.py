from __future__ import absolute_import, division, print_function, unicode_literals
from . import Indicator, MovAv

class DetrendedPriceOscillator(Indicator):
    """
    Defined by Joe DiNapoli in his book *"Trading with DiNapoli levels"*

    It measures the price variations against a Moving Average (the trend)
    and therefore removes the "trend" factor from the price.

    Formula:
      - movav = MovingAverage(close, period)
      - dpo = close - movav(shifted period / 2 + 1)

    See:
      - http://en.wikipedia.org/wiki/Detrended_price_oscillator
    """
    alias = ('DPO',)
    lines = ('dpo',)
    params = (('period', 20), ('movav', MovAv.Simple))
    plotinfo = dict(plothlines=[0.0])

    def _plotlabel(self):
        if False:
            for i in range(10):
                print('nop')
        plabels = [self.p.period]
        plabels += [self.p.movav] * self.p.notdefault('movav')
        return plabels

    def __init__(self):
        if False:
            i = 10
            return i + 15
        ma = self.p.movav(self.data, period=self.p.period)
        self.lines.dpo = self.data - ma(-self.p.period // 2 + 1)
        super(DetrendedPriceOscillator, self).__init__()