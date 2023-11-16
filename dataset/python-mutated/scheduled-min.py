from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt

class St(bt.Strategy):
    params = dict(when=bt.timer.SESSION_START, timer=True, cheat=False, offset=datetime.timedelta(), repeat=datetime.timedelta(), weekdays=[], weekcarry=False, monthdays=[], monthcarry=True)

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        bt.ind.SMA()
        if self.p.timer:
            self.add_timer(when=self.p.when, offset=self.p.offset, repeat=self.p.repeat, weekdays=self.p.weekdays, weekcarry=self.p.weekcarry, monthdays=self.p.monthdays, monthcarry=self.p.monthcarry)
        if self.p.cheat:
            self.add_timer(when=self.p.when, offset=self.p.offset, repeat=self.p.repeat, weekdays=self.p.weekdays, weekcarry=self.p.weekcarry, monthdays=self.p.monthdays, monthcarry=self.p.monthcarry, tzdata=self.data0, cheat=True)
        self.order = None

    def prenext(self):
        if False:
            while True:
                i = 10
        self.next()

    def next(self):
        if False:
            return 10
        (_, isowk, isowkday) = self.datetime.date().isocalendar()
        txt = '{}, {}, Week {}, Day {}, O {}, H {}, L {}, C {}'.format(len(self), self.datetime.datetime(), isowk, isowkday, self.data.open[0], self.data.high[0], self.data.low[0], self.data.close[0])
        print(txt)

    def notify_timer(self, timer, when, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        print('strategy notify_timer with tid {}, when {} cheat {}'.format(timer.p.tid, when, timer.p.cheat))
        if self.order is None and timer.params.cheat:
            print('-- {} Create buy order'.format(self.data.datetime.datetime()))
            self.order = self.buy()

    def notify_order(self, order):
        if False:
            print('Hello World!')
        if order.status == order.Completed:
            print('-- {} Buy Exec @ {}'.format(self.data.datetime.datetime(), order.executed.price))

def runstrat(args=None):
    if False:
        i = 10
        return i + 15
    args = parse_args(args)
    cerebro = bt.Cerebro()
    kwargs = dict(timeframe=bt.TimeFrame.Minutes, compression=5, sessionstart=datetime.time(9, 0), sessionend=datetime.time(17, 30))
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
        while True:
            i = 10
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Timer Test Intraday')
    parser.add_argument('--data0', default='../../datas/2006-min-005.txt', required=False, help='Data to read in')
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