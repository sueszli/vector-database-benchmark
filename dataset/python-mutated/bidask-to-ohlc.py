from __future__ import absolute_import, division, print_function
import argparse
import datetime
import backtrader as bt
import backtrader.feeds as btfeeds

class St(bt.Strategy):

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        print(','.join((str(x) for x in [self.data.datetime.datetime(), self.data.open[0], self.data.high[0], self.data.high[0], self.data.close[0], self.data.volume[0]])))

def runstrat():
    if False:
        while True:
            i = 10
    args = parse_args()
    cerebro = bt.Cerebro()
    data = btfeeds.GenericCSVData(dataname=args.data, dtformat='%d/%m/%y', time=1, open=5, high=5, low=5, close=5, volume=7, openinterest=-1, timeframe=bt.TimeFrame.Ticks)
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Ticks, compression=args.compression)
    cerebro.addstrategy(St)
    cerebro.run()
    if args.plot:
        cerebro.plot(style='bar')

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='BidAsk to OHLC')
    parser.add_argument('--data', required=False, default='../../datas/bidask2.csv', help='Data file to be read in')
    parser.add_argument('--compression', required=False, default=2, type=int, help='How much to compress the bars')
    parser.add_argument('--plot', required=False, action='store_true', help='Plot the vars')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()