from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt

class St(bt.Strategy):
    params = dict()

    def __init__(self):
        if False:
            print('Hello World!')
        for d in self.datas:
            bt.ind.RSI(d)

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        pass

def runstrat(args=None):
    if False:
        while True:
            i = 10
    args = parse_args(args)
    cerebro = bt.Cerebro()
    kwargs = dict()
    (dtfmt, tmfmt) = ('%Y-%m-%d', 'T%H:%M:%S')
    for (a, d) in ((getattr(args, x), x) for x in ['fromdate', 'todate']):
        if a:
            strpfmt = dtfmt + tmfmt * ('T' in a)
            kwargs[d] = datetime.datetime.strptime(a, strpfmt)
    data0 = bt.feeds.BacktraderCSVData(dataname=args.data0, **kwargs)
    fkwargs = dict()
    fkwargs.update(**eval('dict(' + args.renko + ')'))
    if not args.dual:
        data0.addfilter(bt.filters.Renko, **fkwargs)
        cerebro.adddata(data0)
    else:
        cerebro.adddata(data0)
        data1 = data0.clone()
        data1.addfilter(bt.filters.Renko, **fkwargs)
        cerebro.adddata(data1)
    cerebro.broker = bt.brokers.BackBroker(**eval('dict(' + args.broker + ')'))
    cerebro.addsizer(bt.sizers.FixedSize, **eval('dict(' + args.sizer + ')'))
    cerebro.addstrategy(St, **eval('dict(' + args.strat + ')'))
    kwargs = dict(stdstats=False)
    kwargs.update(**eval('dict(' + args.cerebro + ')'))
    cerebro.run(**kwargs)
    if args.plot:
        kwargs = dict(style='candle')
        kwargs.update(**eval('dict(' + args.plot + ')'))
        cerebro.plot(**kwargs)

def parse_args(pargs=None):
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Renko bricks sample')
    parser.add_argument('--data0', default='../../datas/2005-2006-day-001.txt', required=False, help='Data to read in')
    parser.add_argument('--fromdate', required=False, default='', help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')
    parser.add_argument('--todate', required=False, default='', help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')
    parser.add_argument('--cerebro', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--broker', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--sizer', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--strat', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--plot', required=False, default='', nargs='?', const='{}', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--renko', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--dual', required=False, action='store_true', help='put the filter on a second version of the data')
    return parser.parse_args(pargs)
if __name__ == '__main__':
    runstrat()