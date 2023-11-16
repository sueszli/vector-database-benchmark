from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import bisect
import calendar
import datetime
import backtrader as bt

class TheStrategy(bt.Strategy):

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        header = ['Len', 'Name', 'RollName', 'Datetime', 'WeekDay', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']
        print(', '.join(header))

    def next(self):
        if False:
            i = 10
            return i + 15
        txt = list()
        txt.append('%04d' % len(self.data0))
        txt.append('{}'.format(self.data0._dataname))
        txt.append('{}'.format(self.data0._d._dataname))
        txt.append('{}'.format(self.data.datetime.date()))
        txt.append('{}'.format(self.data.datetime.date().strftime('%a')))
        txt.append('{}'.format(self.data.open[0]))
        txt.append('{}'.format(self.data.high[0]))
        txt.append('{}'.format(self.data.low[0]))
        txt.append('{}'.format(self.data.close[0]))
        txt.append('{}'.format(self.data.volume[0]))
        txt.append('{}'.format(self.data.openinterest[0]))
        print(', '.join(txt))

def checkdate(dt, d):
    if False:
        i = 10
        return i + 15
    MONTHS = dict(H=3, M=6, U=9, Z=12)
    M = MONTHS[d._dataname[-2]]
    (centuria, year) = divmod(dt.year, 10)
    decade = centuria * 10
    YCode = int(d._dataname[-1])
    Y = decade + YCode
    if Y < dt.year:
        Y += 10
    exp_day = 21 - (calendar.weekday(Y, M, 1) + 2) % 7
    exp_dt = datetime.datetime(Y, M, exp_day)
    (exp_year, exp_week, _) = exp_dt.isocalendar()
    (dt_year, dt_week, _) = dt.isocalendar()
    return (dt_year, dt_week) == (exp_year, exp_week)

def checkvolume(d0, d1):
    if False:
        print('Hello World!')
    return d0.volume[0] < d1.volume[0]

def runstrat(args=None):
    if False:
        for i in range(10):
            print('nop')
    args = parse_args(args)
    cerebro = bt.Cerebro()
    fcodes = ['199FESXM4', '199FESXU4', '199FESXZ4', '199FESXH5', '199FESXM5']
    store = bt.stores.VChartFile()
    ffeeds = [store.getdata(dataname=x) for x in fcodes]
    rollkwargs = dict()
    if args.checkdate:
        rollkwargs['checkdate'] = checkdate
        if args.checkcondition:
            rollkwargs['checkcondition'] = checkvolume
    if not args.no_cerebro:
        if args.rollover:
            cerebro.rolloverdata(*ffeeds, name='FESX', **rollkwargs)
        else:
            cerebro.chaindata(*ffeeds, name='FESX')
    else:
        drollover = bt.feeds.RollOver(*ffeeds, dataname='FESX', **rollkwargs)
        cerebro.adddata(drollover)
    cerebro.addstrategy(TheStrategy)
    cerebro.run(stdstats=False)
    if args.plot:
        pkwargs = dict(style='bar')
        if args.plot is not True:
            npkwargs = eval('dict(' + args.plot + ')')
            pkwargs.update(npkwargs)
        cerebro.plot(**pkwargs)

def parse_args(pargs=None):
    if False:
        return 10
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample for Roll Over of Futures')
    parser.add_argument('--no-cerebro', required=False, action='store_true', help='Use RollOver Directly')
    parser.add_argument('--rollover', required=False, action='store_true')
    parser.add_argument('--checkdate', required=False, action='store_true', help='Change during expiration week')
    parser.add_argument('--checkcondition', required=False, action='store_true', help='Change when a given condition is met')
    parser.add_argument('--plot', '-p', nargs='?', required=False, metavar='kwargs', const=True, help='Plot the read data applying any kwargs passed\n\nFor example:\n\n  --plot style="candle" (to plot candles)\n')
    if pargs is not None:
        return parser.parse_args(pargs)
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()