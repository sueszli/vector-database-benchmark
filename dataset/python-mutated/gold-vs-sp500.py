from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import scipy.stats
import backtrader as bt

class PearsonR(bt.ind.PeriodN):
    _mindatas = 2
    lines = ('correlation',)
    params = (('period', 20),)

    def next(self):
        if False:
            i = 10
            return i + 15
        (c, p) = scipy.stats.pearsonr(self.data0.get(size=self.p.period), self.data1.get(size=self.p.period))
        self.lines.correlation[0] = c

class MACrossOver(bt.Strategy):
    params = (('ma', bt.ind.MovAv.SMA), ('pd1', 20), ('pd2', 20))

    def __init__(self):
        if False:
            return 10
        ma1 = self.p.ma(self.data0, period=self.p.pd1, subplot=True)
        self.p.ma(self.data1, period=self.p.pd2, plotmaster=ma1)
        PearsonR(self.data0, self.data1)

def runstrat(args=None):
    if False:
        i = 10
        return i + 15
    args = parse_args(args)
    cerebro = bt.Cerebro()
    kwargs = dict()
    (dtfmt, tmfmt) = ('%Y-%m-%d', 'T%H:%M:%S')
    for (a, d) in ((getattr(args, x), x) for x in ['fromdate', 'todate']):
        if a:
            strpfmt = dtfmt + tmfmt * ('T' in a)
            kwargs[d] = datetime.datetime.strptime(a, strpfmt)
    if not args.offline:
        YahooData = bt.feeds.YahooFinanceData
    else:
        YahooData = bt.feeds.YahooFinanceCSVData
    data0 = YahooData(dataname=args.data0, **kwargs)
    cerebro.resampledata(data0, timeframe=bt.TimeFrame.Weeks)
    data1 = YahooData(dataname=args.data1, **kwargs)
    cerebro.resampledata(data1, timeframe=bt.TimeFrame.Weeks)
    data1.plotinfo.plotmaster = data0
    kwargs = eval('dict(' + args.broker + ')')
    cerebro.broker = bt.brokers.BackBroker(**kwargs)
    kwargs = eval('dict(' + args.sizer + ')')
    cerebro.addsizer(bt.sizers.FixedSize, **kwargs)
    if True:
        kwargs = eval('dict(' + args.strat + ')')
        cerebro.addstrategy(MACrossOver, **kwargs)
    cerebro.addobserver(bt.observers.LogReturns2, timeframe=bt.TimeFrame.Weeks, compression=20)
    cerebro.run(**eval('dict(' + args.cerebro + ')'))
    if args.plot:
        cerebro.plot(**eval('dict(' + args.plot + ')'))

def parse_args(pargs=None):
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Gold vs SP500 from https://estrategiastrading.com/oro-bolsa-estadistica-con-python/')
    parser.add_argument('--data0', required=False, default='SPY', metavar='TICKER', help='Yahoo ticker to download')
    parser.add_argument('--data1', required=False, default='GLD', metavar='TICKER', help='Yahoo ticker to download')
    parser.add_argument('--offline', required=False, action='store_true', help='Use the offline files')
    parser.add_argument('--fromdate', required=False, default='2005-01-01', help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')
    parser.add_argument('--todate', required=False, default='2016-01-01', help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')
    parser.add_argument('--cerebro', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--broker', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--sizer', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--strat', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--plot', required=False, default='', nargs='?', const='{}', metavar='kwargs', help='kwargs in key=value format')
    return parser.parse_args(pargs)
if __name__ == '__main__':
    runstrat()