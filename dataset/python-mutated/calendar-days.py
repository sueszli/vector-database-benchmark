from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt
import backtrader.indicators as btind
import backtrader.feeds as btfeeds
import backtrader.filters as btfilters

def runstrat():
    if False:
        while True:
            i = 10
    args = parse_args()
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(bt.Strategy)
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    data = btfeeds.BacktraderCSVData(dataname=args.data, fromdate=fromdate, todate=todate)
    if args.calendar:
        if args.fprice is not None:
            args.fprice = float(args.fprice)
        data.addfilter(btfilters.CalendarDays, fill_price=args.fprice, fill_vol=args.fvol)
    cerebro.adddata(data)
    if args.sma:
        cerebro.addindicator(btind.SMA, period=args.period)
    if args.writer:
        cerebro.addwriter(bt.WriterFile, csv=args.wrcsv)
    cerebro.run()
    if args.plot:
        cerebro.plot(style='bar', numfigs=args.numfigs, volume=False)

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Calendar Days Filter Sample')
    parser.add_argument('--data', '-d', default='../../datas/2006-day-001.txt', help='data to add to the system')
    parser.add_argument('--fromdate', '-f', default='2006-01-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', default='2006-12-31', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--calendar', '-cal', required=False, action='store_true', help='Add a CalendarDays filter')
    parser.add_argument('--fprice', required=False, default=None, help='Use as fill for price (None for previous close)')
    parser.add_argument('--fvol', required=False, default=0.0, type=float, help='Use as fill volume for missing bar (def: 0.0)')
    parser.add_argument('--sma', required=False, action='store_true', help='Add a Simple Moving Average')
    parser.add_argument('--period', default=15, type=int, help='Period to apply to the Simple Moving Average')
    parser.add_argument('--writer', '-w', action='store_true', help='Add a writer to cerebro')
    parser.add_argument('--wrcsv', '-wc', action='store_true', help='Enable CSV Output in the writer')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the read data')
    parser.add_argument('--numfigs', '-n', default=1, help='Plot using numfigs figures')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()