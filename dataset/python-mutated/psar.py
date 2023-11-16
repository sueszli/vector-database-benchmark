from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt

class St(bt.Strategy):
    params = ()

    def __init__(self):
        if False:
            print('Hello World!')
        self.psar = bt.ind.ParabolicSAR(period=20)
        pass

    def next(self):
        if False:
            return 10
        txt = ['{:4d}'.format(len(self))]
        txt.append('{}'.format(self.datetime.date()))
        txt.append('{:.2f}'.format(self.psar[0]))
        print(','.join(txt))

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
    cerebro.adddata(data0)
    cerebro.broker = bt.brokers.BackBroker(**eval('dict(' + args.broker + ')'))
    cerebro.addsizer(bt.sizers.FixedSize, **eval('dict(' + args.sizer + ')'))
    cerebro.addstrategy(St, **eval('dict(' + args.strat + ')'))
    cerebro.run(**eval('dict(' + args.cerebro + ')'))
    if args.plot:
        cerebro.plot(**eval('dict(' + args.plot + ')'))

def parse_args(pargs=None):
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample Skeleton')
    parser.add_argument('--data0', default='../../datas/2005-2006-day-001.txt', required=False, help='Data to read in')
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