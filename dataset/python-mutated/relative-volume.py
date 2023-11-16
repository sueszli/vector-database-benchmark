from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt
import backtrader.feeds as btfeeds
from relvolbybar import RelativeVolumeByBar

def runstrategy():
    if False:
        for i in range(10):
            print('nop')
    args = parse_args()
    cerebro = bt.Cerebro()
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    data = btfeeds.BacktraderCSVData(dataname=args.data, fromdate=fromdate, todate=todate)
    cerebro.adddata(data)
    cerebro.addstrategy(bt.Strategy)
    prestart = datetime.datetime.strptime(args.prestart, '%H:%M').time()
    start = datetime.datetime.strptime(args.start, '%H:%M').time()
    end = datetime.datetime.strptime(args.end, '%H:%M').time()
    cerebro.addindicator(RelativeVolumeByBar, prestart=prestart, start=start, end=end)
    if args.writer:
        cerebro.addwriter(bt.WriterFile, csv=args.wrcsv)
    cerebro.run(stdstats=False)
    if args.plot:
        cerebro.plot(numfigs=args.numfigs, volume=True)

def parse_args():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='MultiData Strategy')
    parser.add_argument('--data', '-d', default='../../datas/2006-01-02-volume-min-001.txt', help='data to add to the system')
    parser.add_argument('--prestart', default='08:00', help='Start time for the Session Filter')
    parser.add_argument('--start', default='09:15', help='Start time for the Session Filter')
    parser.add_argument('--end', '-te', default='17:15', help='End time for the Session Filter')
    parser.add_argument('--fromdate', '-f', default='2006-01-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', default='2006-12-31', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--writer', '-w', action='store_true', help='Add a writer to cerebro')
    parser.add_argument('--wrcsv', '-wc', action='store_true', help='Enable CSV Output in the writer')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the read data')
    parser.add_argument('--numfigs', '-n', default=1, help='Plot using numfigs figures')
    return parser.parse_args()
if __name__ == '__main__':
    runstrategy()