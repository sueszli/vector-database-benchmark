from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import collections
import datetime
import backtrader as bt
MAINSIGNALS = collections.OrderedDict((('longshort', bt.SIGNAL_LONGSHORT), ('longonly', bt.SIGNAL_LONG), ('shortonly', bt.SIGNAL_SHORT)))
EXITSIGNALS = {'longexit': bt.SIGNAL_LONGEXIT, 'shortexit': bt.SIGNAL_LONGEXIT}

class SMACloseSignal(bt.Indicator):
    lines = ('signal',)
    params = (('period', 30),)

    def __init__(self):
        if False:
            while True:
                i = 10
        self.lines.signal = self.data - bt.indicators.SMA(period=self.p.period)

class SMAExitSignal(bt.Indicator):
    lines = ('signal',)
    params = (('p1', 5), ('p2', 30))

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        sma1 = bt.indicators.SMA(period=self.p.p1)
        sma2 = bt.indicators.SMA(period=self.p.p2)
        self.lines.signal = sma1 - sma2

def runstrat(args=None):
    if False:
        for i in range(10):
            print('nop')
    args = parse_args(args)
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(args.cash)
    dkwargs = dict()
    if args.fromdate is not None:
        fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
        dkwargs['fromdate'] = fromdate
    if args.todate is not None:
        todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
        dkwargs['todate'] = todate
    data = bt.feeds.BacktraderCSVData(dataname=args.data, **dkwargs)
    cerebro.adddata(data)
    cerebro.add_signal(MAINSIGNALS[args.signal], SMACloseSignal, period=args.smaperiod)
    if args.exitsignal is not None:
        cerebro.add_signal(EXITSIGNALS[args.exitsignal], SMAExitSignal, p1=args.exitperiod, p2=args.smaperiod)
    cerebro.run()
    if args.plot:
        pkwargs = dict(style='bar')
        if args.plot is not True:
            npkwargs = eval('dict(' + args.plot + ')')
            pkwargs.update(npkwargs)
        cerebro.plot(**pkwargs)

def parse_args(pargs=None):
    if False:
        return 10
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample for Signal concepts')
    parser.add_argument('--data', required=False, default='../../datas/2005-2006-day-001.txt', help='Specific data to be read in')
    parser.add_argument('--fromdate', required=False, default=None, help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', required=False, default=None, help='Ending date in YYYY-MM-DD format')
    parser.add_argument('--cash', required=False, action='store', type=float, default=50000, help='Cash to start with')
    parser.add_argument('--smaperiod', required=False, action='store', type=int, default=30, help='Period for the moving average')
    parser.add_argument('--exitperiod', required=False, action='store', type=int, default=5, help='Period for the exit control SMA')
    parser.add_argument('--signal', required=False, action='store', default=MAINSIGNALS.keys()[0], choices=MAINSIGNALS, help='Signal type to use for the main signal')
    parser.add_argument('--exitsignal', required=False, action='store', default=None, choices=EXITSIGNALS, help='Signal type to use for the exit signal')
    parser.add_argument('--plot', '-p', nargs='?', required=False, metavar='kwargs', const=True, help='Plot the read data applying any kwargs passed\n\nFor example:\n\n  --plot style="candle" (to plot candles)\n')
    if pargs is not None:
        return parser.parse_args(pargs)
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()