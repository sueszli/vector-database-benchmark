from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt

class NYSE_2016(bt.TradingCalendar):
    params = dict(holidays=[datetime.date(2016, 1, 1), datetime.date(2016, 1, 18), datetime.date(2016, 2, 15), datetime.date(2016, 3, 25), datetime.date(2016, 5, 30), datetime.date(2016, 7, 4), datetime.date(2016, 9, 5), datetime.date(2016, 11, 24), datetime.date(2016, 12, 26)], earlydays=[(datetime.date(2016, 11, 25), datetime.time(9, 30), datetime.time(13, 1))], open=datetime.time(9, 30), close=datetime.time(16, 0))

class St(bt.Strategy):
    params = dict()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def prenext(self):
        if False:
            return 10
        self.next()

    def next(self):
        if False:
            print('Hello World!')
        print('Strategy len {} datetime {}'.format(len(self), self.datetime.datetime()), end=' ')
        print('Data0 len {} datetime {}'.format(len(self.data0), self.data0.datetime.datetime()), end=' ')
        if len(self.data1):
            print('Data1 len {} datetime {}'.format(len(self.data1), self.data1.datetime.datetime()))
        else:
            print()

def runstrat(args=None):
    if False:
        i = 10
        return i + 15
    args = parse_args(args)
    cerebro = bt.Cerebro()
    tzinput = 'Europe/Berlin'
    tz = 'US/Eastern'
    kwargs = dict(tzinput=tzinput, tz=tz)
    (dtfmt, tmfmt) = ('%Y-%m-%d', 'T%H:%M:%S')
    for (a, d) in ((getattr(args, x), x) for x in ['fromdate', 'todate']):
        if a:
            strpfmt = dtfmt + tmfmt * ('T' in a)
            kwargs[d] = datetime.datetime.strptime(a, strpfmt)
    data0 = bt.feeds.BacktraderCSVData(dataname=args.data0, **kwargs)
    cerebro.adddata(data0)
    d1 = cerebro.resampledata(data0, timeframe=getattr(bt.TimeFrame, args.timeframe))
    if args.pandascal:
        cerebro.addcalendar(args.pandascal)
    elif args.owncal:
        cerebro.addcalendar(NYSE_2016())
    cerebro.broker = bt.brokers.BackBroker(**eval('dict(' + args.broker + ')'))
    cerebro.addsizer(bt.sizers.FixedSize, **eval('dict(' + args.sizer + ')'))
    cerebro.addstrategy(St, **eval('dict(' + args.strat + ')'))
    cerebro.run(**eval('dict(' + args.cerebro + ')'))
    if args.plot:
        cerebro.plot(**eval('dict(' + args.plot + ')'))

def parse_args(pargs=None):
    if False:
        return 10
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Trading Calendar Sample')
    parser.add_argument('--data0', default='yhoo-2016-11.csv', required=False, help='Data to read in')
    parser.add_argument('--fromdate', required=False, default='2016-01-01', help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')
    parser.add_argument('--todate', required=False, default='2016-12-31', help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')
    parser.add_argument('--cerebro', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--broker', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--sizer', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--strat', required=False, default='', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--plot', required=False, default='', nargs='?', const='{}', metavar='kwargs', help='kwargs in key=value format')
    pgroup = parser.add_mutually_exclusive_group(required=False)
    pgroup.add_argument('--pandascal', required=False, action='store', default='', help='Name of trading calendar to use')
    pgroup.add_argument('--owncal', required=False, action='store_true', help='Apply custom NYSE 2016 calendar')
    parser.add_argument('--timeframe', required=False, action='store', default='Days', choices=['Days'], help='Timeframe to resample to')
    return parser.parse_args(pargs)
if __name__ == '__main__':
    runstrat()