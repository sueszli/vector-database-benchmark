from __future__ import absolute_import, division, print_function, unicode_literals
import math
from backtrader import Analyzer
from backtrader.mathsupport import average, standarddev
from backtrader.utils import AutoOrderedDict

class SQN(Analyzer):
    """SQN or SystemQualityNumber. Defined by Van K. Tharp to categorize trading
    systems.

      - 1.6 - 1.9 Below average
      - 2.0 - 2.4 Average
      - 2.5 - 2.9 Good
      - 3.0 - 5.0 Excellent
      - 5.1 - 6.9 Superb
      - 7.0 -     Holy Grail?

    The formula:

      - SquareRoot(NumberTrades) * Average(TradesProfit) / StdDev(TradesProfit)

    The sqn value should be deemed reliable when the number of trades >= 30

    Methods:

      - get_analysis

        Returns a dictionary with keys "sqn" and "trades" (number of
        considered trades)

    """
    alias = ('SystemQualityNumber',)

    def create_analysis(self):
        if False:
            while True:
                i = 10
        'Replace default implementation to instantiate an AutoOrdereDict\n        rather than an OrderedDict'
        self.rets = AutoOrderedDict()

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        super(SQN, self).start()
        self.pnl = list()
        self.count = 0

    def notify_trade(self, trade):
        if False:
            return 10
        if trade.status == trade.Closed:
            self.pnl.append(trade.pnlcomm)
            self.count += 1

    def stop(self):
        if False:
            i = 10
            return i + 15
        if self.count > 1:
            pnl_av = average(self.pnl)
            pnl_stddev = standarddev(self.pnl)
            try:
                sqn = math.sqrt(len(self.pnl)) * pnl_av / pnl_stddev
            except ZeroDivisionError:
                sqn = None
        else:
            sqn = 0
        self.rets.sqn = sqn
        self.rets.trades = self.count