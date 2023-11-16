from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

class BidAskCSV(btfeeds.GenericCSVData):
    linesoverride = True
    lines = ('bid', 'ask', 'datetime')
    params = (('bid', 1), ('ask', 2))

class St(bt.Strategy):
    params = (('sma', False), ('period', 3))

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.p.sma:
            self.sma = btind.SMA(self.data, period=self.p.period)

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        dtstr = self.data.datetime.datetime().isoformat()
        txt = '%4d: %s - Bid %.4f - %.4f Ask' % (len(self), dtstr, self.data.bid[0], self.data.ask[0])
        if self.p.sma:
            txt += ' - SMA: %.4f' % self.sma[0]
        print(txt)

def parse_args():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='Bid/Ask Line Hierarchy', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', '-d', action='store', required=False, default='../../datas/bidask.csv', help='data to add to the system')
    parser.add_argument('--dtformat', '-dt', required=False, default='%m/%d/%Y %H:%M:%S', help='Format of datetime in input')
    parser.add_argument('--sma', '-s', action='store_true', required=False, help='Add an SMA to the mix')
    parser.add_argument('--period', '-p', action='store', required=False, default=5, type=int, help='Period for the sma')
    return parser.parse_args()

def runstrategy():
    if False:
        return 10
    args = parse_args()
    cerebro = bt.Cerebro()
    data = BidAskCSV(dataname=args.data, dtformat=args.dtformat)
    cerebro.adddata(data)
    cerebro.addstrategy(St, sma=args.sma, period=args.period)
    cerebro.run()
if __name__ == '__main__':
    runstrategy()