from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt

class St(bt.Strategy):
    params = (('ondata', False),)

    def __init__(self):
        if False:
            while True:
                i = 10
        if not self.p.ondata:
            a = self.data.high - self.data.low
        else:
            a = 1.05 * (self.data.high + self.data.low) / 2.0
        b = bt.LinePlotterIndicator(a, name='hilo')
        b.plotinfo.subplot = not self.p.ondata

def runstrat(pargs=None):
    if False:
        return 10
    args = parse_args(pargs)
    cerebro = bt.Cerebro()
    dkwargs = dict()
    if args.fromdate is not None:
        fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
        dkwargs['fromdate'] = fromdate
    if args.todate is not None:
        todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
        dkwargs['todate'] = todate
    data = bt.feeds.BacktraderCSVData(dataname=args.data, **dkwargs)
    cerebro.adddata(data)
    cerebro.addstrategy(St, ondata=args.ondata)
    cerebro.run(stdstats=False)
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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Fake Indicator')
    parser.add_argument('--data', '-d', default='../../datas/2005-2006-day-001.txt', help='data to add to the system')
    parser.add_argument('--fromdate', '-f', default=None, help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', default=None, help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--ondata', '-o', action='store_true', help='Plot fake indicator on the data')
    parser.add_argument('--plot', '-p', nargs='?', required=False, metavar='kwargs', const=True, help='Plot the read data applying any kwargs passed\n\nFor example:\n\n  --plot style="candle" (to plot candles)\n')
    if pargs is not None:
        return parser.parse_args(pargs)
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()