from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import random
import backtrader as bt
BTVERSION = tuple((int(x) for x in bt.__version__.split('.')))

class FixedPerc(bt.Sizer):
    """This sizer simply returns a fixed size for any operation

    Params:
      - ``perc`` (default: ``0.20``) Perc of cash to allocate for operation
    """
    params = (('perc', 0.2),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if False:
            return 10
        cashtouse = self.p.perc * cash
        if BTVERSION > (1, 7, 1, 93):
            size = comminfo.getsize(data.close[0], cashtouse)
        else:
            size = cashtouse // data.close[0]
        return size

class TheStrategy(bt.Strategy):
    """
    This strategy is loosely based on some of the examples from the Van
    K. Tharp book: *Trade Your Way To Financial Freedom*. The logic:

      - Enter the market if:
        - The MACD.macd line crosses the MACD.signal line to the upside
        - The Simple Moving Average has a negative direction in the last x
          periods (actual value below value x periods ago)

     - Set a stop price x times the ATR value away from the close

     - If in the market:

       - Check if the current close has gone below the stop price. If yes,
         exit.
       - If not, update the stop price if the new stop price would be higher
         than the current
    """
    params = (('macd1', 12), ('macd2', 26), ('macdsig', 9), ('atrperiod', 14), ('atrdist', 3.0), ('smaperiod', 30), ('dirperiod', 10))

    def notify_order(self, order):
        if False:
            i = 10
            return i + 15
        if order.status == order.Completed:
            pass
        if not order.alive():
            self.order = None

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.macd = bt.indicators.MACD(self.data, period_me1=self.p.macd1, period_me2=self.p.macd2, period_signal=self.p.macdsig)
        self.mcross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atrperiod)
        self.sma = bt.indicators.SMA(self.data, period=self.p.smaperiod)
        self.smadir = self.sma - self.sma(-self.p.dirperiod)

    def start(self):
        if False:
            while True:
                i = 10
        self.order = None

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        if self.order:
            return
        if not self.position:
            if self.mcross[0] > 0.0 and self.smadir < 0.0:
                self.order = self.buy()
                pdist = self.atr[0] * self.p.atrdist
                self.pstop = self.data.close[0] - pdist
        else:
            pclose = self.data.close[0]
            pstop = self.pstop
            if pclose < pstop:
                self.close()
            else:
                pdist = self.atr[0] * self.p.atrdist
                self.pstop = max(pstop, pclose - pdist)
DATASETS = {'yhoo': '../../datas/yhoo-1996-2014.txt', 'orcl': '../../datas/orcl-1995-2014.txt', 'nvda': '../../datas/nvda-1999-2014.txt'}

def runstrat(args=None):
    if False:
        while True:
            i = 10
    args = parse_args(args)
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(args.cash)
    comminfo = bt.commissions.CommInfo_Stocks_Perc(commission=args.commperc, percabs=True)
    cerebro.broker.addcommissioninfo(comminfo)
    dkwargs = dict()
    if args.fromdate is not None:
        fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
        dkwargs['fromdate'] = fromdate
    if args.todate is not None:
        todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
        dkwargs['todate'] = todate
    dataname = DATASETS.get(args.dataset, args.data)
    data0 = bt.feeds.YahooFinanceCSVData(dataname=dataname, **dkwargs)
    cerebro.adddata(data0)
    cerebro.addstrategy(TheStrategy, macd1=args.macd1, macd2=args.macd2, macdsig=args.macdsig, atrperiod=args.atrperiod, atrdist=args.atrdist, smaperiod=args.smaperiod, dirperiod=args.dirperiod)
    cerebro.addsizer(FixedPerc, perc=args.cashalloc)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='alltime_roi', timeframe=bt.TimeFrame.NoTimeFrame)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, data=data0, _name='benchmark', timeframe=bt.TimeFrame.NoTimeFrame)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Years)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, timeframe=bt.TimeFrame.Years, riskfreerate=args.riskfreerate)
    cerebro.addanalyzer(bt.analyzers.SQN)
    cerebro.addobserver(bt.observers.DrawDown)
    results = cerebro.run()
    st0 = results[0]
    for alyzer in st0.analyzers:
        alyzer.print()
    if args.plot:
        pkwargs = dict(style='bar')
        if args.plot is not True:
            npkwargs = eval('dict(' + args.plot + ')')
            pkwargs.update(npkwargs)
        cerebro.plot(**pkwargs)

def parse_args(pargs=None):
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample for Tharp example with MACD')
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--data', required=False, default=None, help='Specific data to be read in')
    group1.add_argument('--dataset', required=False, action='store', default=None, choices=DATASETS.keys(), help='Choose one of the predefined data sets')
    parser.add_argument('--fromdate', required=False, default='2005-01-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', required=False, default=None, help='Ending date in YYYY-MM-DD format')
    parser.add_argument('--cash', required=False, action='store', type=float, default=50000, help='Cash to start with')
    parser.add_argument('--cashalloc', required=False, action='store', type=float, default=0.2, help='Perc (abs) of cash to allocate for ops')
    parser.add_argument('--commperc', required=False, action='store', type=float, default=0.0033, help='Perc (abs) commision in each operation. 0.001 -> 0.1%%, 0.01 -> 1%%')
    parser.add_argument('--macd1', required=False, action='store', type=int, default=12, help='MACD Period 1 value')
    parser.add_argument('--macd2', required=False, action='store', type=int, default=26, help='MACD Period 2 value')
    parser.add_argument('--macdsig', required=False, action='store', type=int, default=9, help='MACD Signal Period value')
    parser.add_argument('--atrperiod', required=False, action='store', type=int, default=14, help='ATR Period To Consider')
    parser.add_argument('--atrdist', required=False, action='store', type=float, default=3.0, help='ATR Factor for stop price calculation')
    parser.add_argument('--smaperiod', required=False, action='store', type=int, default=30, help='Period for the moving average')
    parser.add_argument('--dirperiod', required=False, action='store', type=int, default=10, help='Period for SMA direction calculation')
    parser.add_argument('--riskfreerate', required=False, action='store', type=float, default=0.01, help='Risk free rate in Perc (abs) of the asset for the Sharpe Ratio')
    parser.add_argument('--plot', '-p', nargs='?', required=False, metavar='kwargs', const=True, help='Plot the read data applying any kwargs passed\n\nFor example:\n\n  --plot style="candle" (to plot candles)\n')
    if pargs is not None:
        return parser.parse_args(pargs)
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()