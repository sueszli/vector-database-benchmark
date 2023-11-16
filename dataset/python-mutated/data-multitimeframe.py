from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
from backtrader import ResamplerDaily, ResamplerWeekly, ResamplerMonthly
from backtrader import ReplayerDaily, ReplayerWeekly, ReplayerMonthly
from backtrader.utils import flushfile

class SMAStrategy(bt.Strategy):
    params = (('period', 10), ('onlydaily', False))

    def __init__(self):
        if False:
            return 10
        self.sma_small_tf = btind.SMA(self.data, period=self.p.period)
        bt.indicators.MACD(self.data0)
        if not self.p.onlydaily:
            self.sma_large_tf = btind.SMA(self.data1, period=self.p.period)
            bt.indicators.MACD(self.data1)

    def prenext(self):
        if False:
            for i in range(10):
                print('nop')
        self.next()

    def nextstart(self):
        if False:
            return 10
        print('--------------------------------------------------')
        print('nextstart called with len', len(self))
        print('--------------------------------------------------')
        super(SMAStrategy, self).nextstart()

    def next(self):
        if False:
            print('Hello World!')
        print('Strategy:', len(self))
        txt = list()
        txt.append('Data0')
        txt.append('%04d' % len(self.data0))
        dtfmt = '%Y-%m-%dT%H:%M:%S.%f'
        txt.append('{:f}'.format(self.data.datetime[0]))
        txt.append('%s' % self.data.datetime.datetime(0).strftime(dtfmt))
        txt.append('{:f}'.format(self.data.close[0]))
        print(', '.join(txt))
        if len(self.datas) > 1 and len(self.data1):
            txt = list()
            txt.append('Data1')
            txt.append('%04d' % len(self.data1))
            dtfmt = '%Y-%m-%dT%H:%M:%S.%f'
            txt.append('{:f}'.format(self.data1.datetime[0]))
            txt.append('%s' % self.data1.datetime.datetime(0).strftime(dtfmt))
            txt.append('{}'.format(self.data1.close[0]))
            print(', '.join(txt))

def runstrat():
    if False:
        for i in range(10):
            print('nop')
    args = parse_args()
    cerebro = bt.Cerebro()
    if not args.indicators:
        cerebro.addstrategy(bt.Strategy)
    else:
        cerebro.addstrategy(SMAStrategy, period=args.period, onlydaily=args.onlydaily)
    datapath = args.dataname or '../../datas/2006-day-001.txt'
    data = btfeeds.BacktraderCSVData(dataname=datapath)
    tframes = dict(daily=bt.TimeFrame.Days, weekly=bt.TimeFrame.Weeks, monthly=bt.TimeFrame.Months)
    if args.noresample:
        datapath = args.dataname2 or '../../datas/2006-week-001.txt'
        data2 = btfeeds.BacktraderCSVData(dataname=datapath)
    elif args.oldrs:
        if args.replay:
            data2 = bt.DataReplayer(dataname=data, timeframe=tframes[args.timeframe], compression=args.compression)
        else:
            data2 = bt.DataResampler(dataname=data, timeframe=tframes[args.timeframe], compression=args.compression)
    else:
        data2 = bt.DataClone(dataname=data)
        if args.replay:
            if args.timeframe == 'daily':
                data2.addfilter(ReplayerDaily)
            elif args.timeframe == 'weekly':
                data2.addfilter(ReplayerWeekly)
            elif args.timeframe == 'monthly':
                data2.addfilter(ReplayerMonthly)
        elif args.timeframe == 'daily':
            data2.addfilter(ResamplerDaily)
        elif args.timeframe == 'weekly':
            data2.addfilter(ResamplerWeekly)
        elif args.timeframe == 'monthly':
            data2.addfilter(ResamplerMonthly)
    cerebro.adddata(data)
    cerebro.adddata(data2)
    cerebro.run(runonce=not args.runnext, preload=not args.nopreload, oldsync=args.oldsync, stdstats=False)
    if args.plot:
        cerebro.plot(style='bar')

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Pandas test script')
    parser.add_argument('--dataname', default='', required=False, help='File Data to Load')
    parser.add_argument('--dataname2', default='', required=False, help='Larger timeframe file to load')
    parser.add_argument('--runnext', action='store_true', help='Use next by next instead of runonce')
    parser.add_argument('--nopreload', action='store_true', help='Do not preload the data')
    parser.add_argument('--oldsync', action='store_true', help='Use old data synchronization method')
    parser.add_argument('--oldrs', action='store_true', help='Use old resampler')
    parser.add_argument('--replay', action='store_true', help='Replay instead of resample')
    parser.add_argument('--noresample', action='store_true', help='Do not resample, rather load larger timeframe')
    parser.add_argument('--timeframe', default='weekly', required=False, choices=['daily', 'weekly', 'monthly'], help='Timeframe to resample to')
    parser.add_argument('--compression', default=1, required=False, type=int, help='Compress n bars into 1')
    parser.add_argument('--indicators', action='store_true', help='Wether to apply Strategy with indicators')
    parser.add_argument('--onlydaily', action='store_true', help='Indicator only to be applied to daily timeframe')
    parser.add_argument('--period', default=10, required=False, type=int, help='Period to apply to indicator')
    parser.add_argument('--plot', required=False, action='store_true', help='Plot the chart')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()