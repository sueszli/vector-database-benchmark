from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

class SMAStrategy(bt.Strategy):
    params = (('period', 10), ('onlydaily', False))

    def __init__(self):
        if False:
            while True:
                i = 10
        self.sma = btind.SMA(self.data, period=self.p.period)

    def start(self):
        if False:
            print('Hello World!')
        self.counter = 0

    def prenext(self):
        if False:
            i = 10
            return i + 15
        self.counter += 1
        print('prenext len %d - counter %d' % (len(self), self.counter))

    def next(self):
        if False:
            print('Hello World!')
        self.counter += 1
        print('---next len %d - counter %d' % (len(self), self.counter))

def runstrat():
    if False:
        for i in range(10):
            print('nop')
    args = parse_args()
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(SMAStrategy, period=args.period)
    datapath = args.dataname or '../../datas//2006-day-001.txt'
    data = btfeeds.BacktraderCSVData(dataname=datapath)
    tframes = dict(daily=bt.TimeFrame.Days, weekly=bt.TimeFrame.Weeks, monthly=bt.TimeFrame.Months)
    if args.oldrp:
        data = bt.DataReplayer(dataname=data, timeframe=tframes[args.timeframe], compression=args.compression)
    else:
        data.replay(timeframe=tframes[args.timeframe], compression=args.compression)
    cerebro.adddata(data)
    cerebro.run(preload=False)
    cerebro.plot(style='bar')

def parse_args():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Pandas test script')
    parser.add_argument('--dataname', default='', required=False, help='File Data to Load')
    parser.add_argument('--oldrp', required=False, action='store_true', help='Use deprecated DataReplayer')
    parser.add_argument('--timeframe', default='weekly', required=False, choices=['daily', 'weekly', 'monthly'], help='Timeframe to resample to')
    parser.add_argument('--compression', default=1, required=False, type=int, help='Compress n bars into 1')
    parser.add_argument('--period', default=10, required=False, type=int, help='Period to apply to indicator')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()