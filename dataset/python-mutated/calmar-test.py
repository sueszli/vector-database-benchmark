from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt

class St(bt.SignalStrategy):
    params = ()

    def __init__(self):
        if False:
            print('Hello World!')
        (ma1, ma2) = (bt.ind.SMA(period=15), bt.ind.SMA(period=50))
        self.signal_add(bt.signal.SIGNAL_LONG, bt.ind.CrossOver(ma1, ma2))

    def next2(self):
        if False:
            while True:
                i = 10
        pass

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
    data0 = bt.feeds.YahooFinanceCSVData(dataname=args.data0, **kwargs)
    cerebro.adddata(data0)
    cerebro.broker = bt.brokers.BackBroker(**eval('dict(' + args.broker + ')'))
    cerebro.addanalyzer(bt.analyzers.Calmar)
    cerebro.addsizer(bt.sizers.FixedSize, **eval('dict(' + args.sizer + ')'))
    cerebro.addstrategy(St, **eval('dict(' + args.strat + ')'))
    st0 = cerebro.run(**eval('dict(' + args.cerebro + ')'))[0]
    i = 1
    for (k, v) in st0.analyzers.calmar.get_analysis().items():
        print(i, ': '.join((str(k), str(v))))
        i += 1
    if args.plot:
        cerebro.plot(**eval('dict(' + args.plot + ')'))

def parse_args(pargs=None):
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample Skeleton')
    parser.add_argument('--data0', default='../../datas/orcl-1995-2014.txt', required=False, help='Data to read in')
    parser.add_argument('--fromdate', required=False, default='', help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')
    parser.add_argument('--todate', required=False, default='', help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')
    parser.add_argument('--cerebro', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--broker', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--sizer', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--strat', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--plot', required=False, default='', nargs='?', const='{}', metavar='kwargs', help='kwargs in key=value format')
    return parser.parse_args(pargs)
if __name__ == '__main__':
    runstrat()