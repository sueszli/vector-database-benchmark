from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
from orderobserver import OrderObserver

class MyStrategy(bt.Strategy):
    params = (('smaperiod', 15), ('limitperc', 1.0), ('valid', 7))

    def log(self, txt, dt=None):
        if False:
            for i in range(10):
                print('nop')
        ' Logging function fot this strategy'
        dt = dt or self.data.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if False:
            return 10
        if order.status in [order.Submitted, order.Accepted]:
            self.log('ORDER ACCEPTED/SUBMITTED', dt=order.created.dt)
            self.order = order
            return
        if order.status in [order.Expired]:
            self.log('BUY EXPIRED')
        elif order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (order.executed.price, order.executed.value, order.executed.comm))
            else:
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (order.executed.price, order.executed.value, order.executed.comm))
        self.order = None

    def __init__(self):
        if False:
            i = 10
            return i + 15
        sma = btind.SMA(period=self.p.smaperiod)
        self.buysell = btind.CrossOver(self.data.close, sma, plot=True)
        self.order = None

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        if self.order:
            return
        if self.position:
            if self.buysell < 0:
                self.log('SELL CREATE, %.2f' % self.data.close[0])
                self.sell()
        elif self.buysell > 0:
            plimit = self.data.close[0] * (1.0 - self.p.limitperc / 100.0)
            valid = self.data.datetime.date(0) + datetime.timedelta(days=self.p.valid)
            self.log('BUY CREATE, %.2f' % plimit)
            self.buy(exectype=bt.Order.Limit, price=plimit, valid=valid)

def runstrat():
    if False:
        i = 10
        return i + 15
    cerebro = bt.Cerebro()
    data = bt.feeds.BacktraderCSVData(dataname='../../datas/2006-day-001.txt')
    cerebro.adddata(data)
    cerebro.addobserver(OrderObserver)
    cerebro.addstrategy(MyStrategy)
    cerebro.run()
    cerebro.plot()
if __name__ == '__main__':
    runstrat()