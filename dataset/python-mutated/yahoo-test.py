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
    data = btfeeds.YahooFinanceData(dataname=args.data, fromdate=fromdate, todate=todate)
    cerebro.adddata(data)
    cerebro.addindicator(btind.SMA, period=args.period)
    if args.writer:
        cerebro.addwriter(bt.WriterFile, csv=args.wrcsv)
    cerebro.run()
    if args.plot:
        cerebro.plot(style='bar', numfigs=args.numfigs, volume=False)

def parse_args():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Calendar Days Filter Sample')
    parser.add_argument('--data', '-d', default='YHOO', help='Ticker to download from Yahoo')
    parser.add_argument('--fromdate', '-f', default='2006-01-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', default='2006-12-31', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--period', default=15, type=int, help='Period to apply to the Simple Moving Average')
    parser.add_argument('--writer', '-w', action='store_true', help='Add a writer to cerebro')
    parser.add_argument('--wrcsv', '-wc', action='store_true', help='Enable CSV Output in the writer')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the read data')
    parser.add_argument('--numfigs', '-n', default=1, type=int, help='Plot using numfigs figures')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()