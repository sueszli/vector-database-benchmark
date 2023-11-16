from __future__ import absolute_import, division, print_function, unicode_literals
import backtrader as bt
import os
import sys
import datetime

class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        if False:
            i = 10
            return i + 15
        ' Logging function for this strategy'
        dt = dt or self.datas[0].datetime.date(0)
        print('日志 ： %s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        if False:
            while True:
                i = 10
        print('datas', self.datas)
        self.dataclose_change = self.datas[0].close
        print('init', self.datas[0])
        self.order = None

    def notify_order(self, order):
        if False:
            i = 10
            return i + 15
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def log(self, txt, dt=None):
        if False:
            print('Hello World!')
        dt = dt or self.datas[0].datetime.date(0)
        print('-1', self.datas[0].datetime.date(-1))
        print(' 0', dt)
        print('dataclose_change ', self.dataclose_change[0])

    def next(self):
        if False:
            while True:
                i = 10
        if self.order:
            return
        if not self.position:
            if self.dataclose_change[0] < self.dataclose_change[-1]:
                if self.dataclose_change[-1] < self.dataclose_change[-2]:
                    self.log('BUY CREATE, %.2f' % self.dataclose_change[0])
                    self.order = self.buy()
        elif len(self) >= self.bar_executed + 5:
            self.log('SELL CREATE, %.2f' % self.dataclose_change[0])
            self.order = self.sell()

def main():
    if False:
        i = 10
        return i + 15
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy)
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, '/home/xda/othergit/backtrader/datas/nvda-1999-2014.txt')
    print(modpath)
    data = bt.feeds.YahooFinanceCSVData(dataname=datapath, fromdate=datetime.datetime(2014, 1, 2), todate=datetime.datetime(2014, 9, 1), reverse=False)
    cerebro.adddata(data)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
if __name__ == '__main__':
    main()