from __future__ import absolute_import, division, print_function, unicode_literals
import backtrader as bt
import backtrader.indicators as btind

class MA_CrossOver(bt.Strategy):
    """This is a long-only strategy which operates on a moving average cross

    Note:
      - Although the default

    Buy Logic:
      - No position is open on the data

      - The ``fast`` moving averagecrosses over the ``slow`` strategy to the
        upside.

    Sell Logic:
      - A position exists on the data

      - The ``fast`` moving average crosses over the ``slow`` strategy to the
        downside

    Order Execution Type:
      - Market

    """
    alias = ('SMA_CrossOver',)
    params = (('fast', 10), ('slow', 30), ('_movav', btind.MovAv.SMA))

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        sma_fast = self.p._movav(period=self.p.fast)
        sma_slow = self.p._movav(period=self.p.slow)
        self.buysig = btind.CrossOver(sma_fast, sma_slow)

    def next(self):
        if False:
            print('Hello World!')
        if self.position.size:
            if self.buysig < 0:
                self.sell()
        elif self.buysig > 0:
            self.buy()