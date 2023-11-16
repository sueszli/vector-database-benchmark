from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import math
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.utils.flushfile
import backtrader.filters as btfilters
from relativevolume import RelativeVolume

def runstrategy():
    if False:
        i = 10
        return i + 15
    args = parse_args()
    cerebro = bt.Cerebro()
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    dtstart = datetime.datetime.strptime(args.tstart, '%H:%M')
    dtend = datetime.datetime.strptime(args.tend, '%H:%M')
    data = btfeeds.BacktraderCSVData(dataname=args.data, fromdate=fromdate, todate=todate, timeframe=bt.TimeFrame.Minutes, compression=1, sessionstart=dtstart, sessionend=dtend)
    if args.filter:
        data.addfilter(btfilters.SessionFilter)
    if args.filler:
        data.addfilter(btfilters.SessionFiller, fill_vol=args.fvol)
    cerebro.adddata(data)
    if args.relvol:
        td = (dtend - dtstart).seconds // 60 + 1
        cerebro.addindicator(RelativeVolume, period=td, volisnan=math.isnan(args.fvol))
    cerebro.addstrategy(bt.Strategy)
    if args.writer:
        cerebro.addwriter(bt.WriterFile, csv=args.wrcsv)
    cerebro.run(stdstats=False)
    if args.plot:
        cerebro.plot(numfigs=args.numfigs, volume=True)

def parse_args():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='DataFilter/DataFiller Sample')
    parser.add_argument('--data', '-d', default='../../datas/2006-01-02-volume-min-001.txt', help='data to add to the system')
    parser.add_argument('--filter', '-ft', action='store_true', help='Filter using session start/end times')
    parser.add_argument('--filler', '-fl', action='store_true', help='Fill missing bars inside start/end times')
    parser.add_argument('--fvol', required=False, default=0.0, type=float, help='Use as fill volume for missing bar (def: 0.0)')
    parser.add_argument('--tstart', '-ts', default='09:15', help='Start time for the Session Filter (HH:MM)')
    parser.add_argument('--tend', '-te', default='17:15', help='End time for the Session Filter (HH:MM)')
    parser.add_argument('--relvol', '-rv', action='store_true', help='Add relative volume indicator')
    parser.add_argument('--fromdate', '-f', default='2006-01-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', default='2006-12-31', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--writer', '-w', action='store_true', help='Add a writer to cerebro')
    parser.add_argument('--wrcsv', '-wc', action='store_true', help='Enable CSV Output in the writer')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the read data')
    parser.add_argument('--numfigs', '-n', default=1, help='Plot using numfigs figures')
    return parser.parse_args()
if __name__ == '__main__':
    runstrategy()