from __future__ import absolute_import, division, print_function, unicode_literals
import time
try:
    time_clock = time.process_time
except:
    time_clock = time.clock
import testcommon
import backtrader as bt
import backtrader.indicators as btind
from backtrader.utils.py3 import PY2

class TestStrategy(bt.Strategy):
    params = (('period', 15), ('printdata', True), ('printops', True), ('stocklike', True))

    def log(self, txt, dt=None, nodate=False):
        if False:
            i = 10
            return i + 15
        if not nodate:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print('%s, %s' % (dt.isoformat(), txt))
        else:
            print('---------- %s' % txt)

    def notify_order(self, order):
        if False:
            print('Hello World!')
        if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return
        if order.status == order.Completed:
            if isinstance(order, bt.BuyOrder):
                if self.p.printops:
                    txt = 'BUY, %.2f' % order.executed.price
                    self.log(txt, order.executed.dt)
                chkprice = '%.2f' % order.executed.price
                self.buyexec.append(chkprice)
            else:
                if self.p.printops:
                    txt = 'SELL, %.2f' % order.executed.price
                    self.log(txt, order.executed.dt)
                chkprice = '%.2f' % order.executed.price
                self.sellexec.append(chkprice)
        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            if self.p.printops:
                self.log('%s ,' % order.Status[order.status])
        self.orderid = None

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.orderid = None
        self.sma = btind.SMA(self.data, period=self.p.period)
        self.cross = btind.CrossOver(self.data.close, self.sma, plot=True)

    def start(self):
        if False:
            print('Hello World!')
        if not self.p.stocklike:
            self.broker.setcommission(commission=2.0, mult=10.0, margin=1000.0)
        if self.p.printdata:
            self.log('-------------------------', nodate=True)
            self.log('Starting portfolio value: %.2f' % self.broker.getvalue(), nodate=True)
        self.tstart = time_clock()
        self.buycreate = list()
        self.sellcreate = list()
        self.buyexec = list()
        self.sellexec = list()

    def stop(self):
        if False:
            return 10
        tused = time_clock() - self.tstart
        if self.p.printdata:
            self.log('Time used: %s' % str(tused))
            self.log('Final portfolio value: %.2f' % self.broker.getvalue())
            self.log('Final cash value: %.2f' % self.broker.getcash())
            self.log('-------------------------')
        else:
            pass

    def next(self):
        if False:
            print('Hello World!')
        if self.p.printdata:
            self.log('Open, High, Low, Close, %.2f, %.2f, %.2f, %.2f, Sma, %f' % (self.data.open[0], self.data.high[0], self.data.low[0], self.data.close[0], self.sma[0]))
            self.log('Close %.2f - Sma %.2f' % (self.data.close[0], self.sma[0]))
        if self.orderid:
            return
        if not self.position.size:
            if self.cross > 0.0:
                if self.p.printops:
                    self.log('BUY CREATE , %.2f' % self.data.close[0])
                self.orderid = self.buy()
                chkprice = '%.2f' % self.data.close[0]
                self.buycreate.append(chkprice)
        elif self.cross < 0.0:
            if self.p.printops:
                self.log('SELL CREATE , %.2f' % self.data.close[0])
            self.orderid = self.close()
            chkprice = '%.2f' % self.data.close[0]
            self.sellcreate.append(chkprice)
chkdatas = 1

def test_run(main=False):
    if False:
        i = 10
        return i + 15
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    cerebros = testcommon.runtest(datas, TestStrategy, printdata=main, stocklike=False, printops=main, plot=main, analyzer=(bt.analyzers.TimeReturn, dict(timeframe=bt.TimeFrame.Years)))
    for cerebro in cerebros:
        strat = cerebro.runstrats[0][0]
        analyzer = strat.analyzers[0]
        analysis = analyzer.get_analysis()
        if main:
            print(analysis)
            print(str(analysis[next(iter(analysis.keys()))]))
        else:
            if PY2:
                sval = '0.2795'
            else:
                sval = '0.2794999999999983'
            assert str(analysis[next(iter(analysis.keys()))]) == sval
if __name__ == '__main__':
    test_run(main=True)