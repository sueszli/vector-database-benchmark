from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import random
import backtrader as bt

class CloseSMA(bt.Strategy):
    params = (('period', 15),)

    def __init__(self):
        if False:
            while True:
                i = 10
        sma = bt.indicators.SMA(self.data, period=self.p.period)
        self.crossover = bt.indicators.CrossOver(self.data, sma)

    def next(self):
        if False:
            print('Hello World!')
        if self.crossover > 0:
            self.buy()
        elif self.crossover < 0:
            self.sell()

class LongOnly(bt.Sizer):
    params = (('stake', 1),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if False:
            for i in range(10):
                print('nop')
        if isbuy:
            return self.p.stake
        position = self.broker.getposition(data)
        if not position.size:
            return 0
        return self.p.stake

class FixedReverser(bt.Sizer):
    params = (('stake', 1),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if False:
            i = 10
            return i + 15
        position = self.strategy.getposition(data)
        size = self.p.stake * (1 + (position.size != 0))
        return size

def runstrat(args=None):
    if False:
        i = 10
        return i + 15
    args = parse_args(args)
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(args.cash)
    dkwargs = dict()
    if args.fromdate:
        fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
        dkwargs['fromdate'] = fromdate
    if args.todate:
        todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
        dkwargs['todate'] = todate
    data0 = bt.feeds.YahooFinanceCSVData(dataname=args.data0, **dkwargs)
    cerebro.adddata(data0, name='Data0')
    cerebro.addstrategy(CloseSMA, period=args.period)
    if args.longonly:
        cerebro.addsizer(LongOnly, stake=args.stake)
    else:
        cerebro.addsizer(bt.sizers.FixedReverser, stake=args.stake)
    cerebro.run()
    if args.plot:
        pkwargs = dict()
        if args.plot is not True:
            pkwargs = eval('dict(' + args.plot + ')')
        cerebro.plot(**pkwargs)

def parse_args(pargs=None):
    if False:
        return 10
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample for sizer')
    parser.add_argument('--data0', required=False, default='../../datas/yhoo-1996-2015.txt', help='Data to be read in')
    parser.add_argument('--fromdate', required=False, default='2005-01-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', required=False, default='2006-12-31', help='Ending date in YYYY-MM-DD format')
    parser.add_argument('--cash', required=False, action='store', type=float, default=50000, help='Cash to start with')
    parser.add_argument('--longonly', required=False, action='store_true', help='Use the LongOnly sizer')
    parser.add_argument('--stake', required=False, action='store', type=int, default=1, help='Stake to pass to the sizers')
    parser.add_argument('--period', required=False, action='store', type=int, default=15, help='Period for the Simple Moving Average')
    parser.add_argument('--plot', '-p', nargs='?', required=False, metavar='kwargs', const=True, help='Plot the read data applying any kwargs passed\n\nFor example:\n\n  --plot style="candle" (to plot candles)\n')
    if pargs is not None:
        return parser.parse_args(pargs)
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()