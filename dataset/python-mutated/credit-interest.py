from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import collections
import datetime
import itertools
import backtrader as bt

class SMACrossOver(bt.Signal):
    params = (('p1', 10), ('p2', 30))

    def __init__(self):
        if False:
            return 10
        sma1 = bt.indicators.SMA(period=self.p.p1)
        sma2 = bt.indicators.SMA(period=self.p.p2)
        self.lines.signal = bt.indicators.CrossOver(sma1, sma2)

class NoExit(bt.Signal):

    def next(self):
        if False:
            i = 10
            return i + 15
        self.lines.signal[0] = 0.0

class St(bt.SignalStrategy):
    opcounter = itertools.count(1)

    def notify_order(self, order):
        if False:
            for i in range(10):
                print('nop')
        if order.status == bt.Order.Completed:
            t = ''
            t += '{:02d}'.format(next(self.opcounter))
            t += ' {}'.format(order.data.datetime.datetime())
            t += ' BUY ' * order.isbuy() or ' SELL'
            t += ' Size: {:+d} / Price: {:.2f}'
            print(t.format(order.executed.size, order.executed.price))

    def notify_trade(self, trade):
        if False:
            while True:
                i = 10
        if trade.isclosed:
            print('Trade closed with P&L: Gross {} Net {}'.format(trade.pnl, trade.pnlcomm))

def runstrat(args=None):
    if False:
        while True:
            i = 10
    args = parse_args(args)
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(args.cash)
    cerebro.broker.set_int2pnl(args.no_int2pnl)
    dkwargs = dict()
    if args.fromdate is not None:
        fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
        dkwargs['fromdate'] = fromdate
    if args.todate is not None:
        todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
        dkwargs['todate'] = todate
    data = bt.feeds.BacktraderCSVData(dataname=args.data, **dkwargs)
    cerebro.adddata(data)
    cerebro.signal_strategy(St)
    cerebro.addsizer(bt.sizers.FixedSize, stake=args.stake)
    sigtype = bt.signal.SIGNAL_LONGSHORT
    if args.long:
        sigtype = bt.signal.SIGNAL_LONG
    elif args.short:
        sigtype = bt.signal.SIGNAL_SHORT
    cerebro.add_signal(sigtype, SMACrossOver, p1=args.period1, p2=args.period2)
    if args.no_exit:
        if args.long:
            cerebro.add_signal(bt.signal.SIGNAL_LONGEXIT, NoExit)
        elif args.short:
            cerebro.add_signal(bt.signal.SIGNAL_SHORTEXIT, NoExit)
    comminfo = bt.CommissionInfo(mult=args.mult, margin=args.margin, stocklike=args.stocklike, interest=args.interest, interest_long=args.interest_long)
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.run()
    if args.plot:
        pkwargs = dict(style='bar')
        if args.plot is not True:
            npkwargs = eval('dict(' + args.plot + ')')
            pkwargs.update(npkwargs)
        cerebro.plot(**pkwargs)

def parse_args(pargs=None):
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample for Slippage')
    parser.add_argument('--data', required=False, default='../../datas/2005-2006-day-001.txt', help='Specific data to be read in')
    parser.add_argument('--fromdate', required=False, default=None, help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', required=False, default=None, help='Ending date in YYYY-MM-DD format')
    parser.add_argument('--cash', required=False, action='store', type=float, default=50000, help='Cash to start with')
    parser.add_argument('--period1', required=False, action='store', type=int, default=10, help='Fast moving average period')
    parser.add_argument('--period2', required=False, action='store', type=int, default=30, help='Slow moving average period')
    parser.add_argument('--interest', required=False, action='store', default=0.0, type=float, help='Activate credit interest rate')
    parser.add_argument('--no-int2pnl', required=False, action='store_false', help='Do not assign interest to pnl')
    parser.add_argument('--interest_long', required=False, action='store_true', help='Credit interest rate for long positions')
    pgroup = parser.add_mutually_exclusive_group()
    pgroup.add_argument('--long', required=False, action='store_true', help='Do a long only strategy')
    pgroup.add_argument('--short', required=False, action='store_true', help='Do a long only strategy')
    parser.add_argument('--no-exit', required=False, action='store_true', help='The 1st taken position will not be exited')
    parser.add_argument('--stocklike', required=False, action='store_true', help='Consider the asset to be stocklike')
    parser.add_argument('--margin', required=False, action='store', default=0.0, type=float, help='Margin for future like instruments')
    parser.add_argument('--mult', required=False, action='store', default=1.0, type=float, help='Multiplier for future like instruments')
    parser.add_argument('--stake', required=False, action='store', default=10, type=int, help='Stake to apply')
    parser.add_argument('--plot', '-p', nargs='?', required=False, metavar='kwargs', const=True, help='Plot the read data applying any kwargs passed\n\nFor example:\n\n  --plot style="candle" (to plot candles)\n')
    if pargs is not None:
        return parser.parse_args(pargs)
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()