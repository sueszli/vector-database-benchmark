from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import backtrader as bt
import backtrader.feeds as btfeeds

def runstrat():
    if False:
        return 10
    args = parse_args()
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(bt.Strategy)
    datapath = args.dataname or '../../datas/2006-day-001.txt'
    data = btfeeds.BacktraderCSVData(dataname=datapath)
    tframes = dict(daily=bt.TimeFrame.Days, weekly=bt.TimeFrame.Weeks, monthly=bt.TimeFrame.Months)
    if args.oldrs:
        data = bt.DataResampler(dataname=data, timeframe=tframes[args.timeframe], compression=args.compression)
        cerebro.adddata(data)
    else:
        cerebro.resampledata(data, timeframe=tframes[args.timeframe], compression=args.compression)
    cerebro.run()
    cerebro.plot(style='bar')

def parse_args():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Resample down to minutes')
    parser.add_argument('--dataname', default='', required=False, help='File Data to Load')
    parser.add_argument('--oldrs', required=False, action='store_true', help='Use deprecated DataResampler')
    parser.add_argument('--timeframe', default='weekly', required=False, choices=['daily', 'weekly', 'monthly'], help='Timeframe to resample to')
    parser.add_argument('--compression', default=1, required=False, type=int, help='Compress n bars into 1')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()