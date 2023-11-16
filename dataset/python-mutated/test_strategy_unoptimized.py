from __future__ import absolute_import, division, print_function, unicode_literals
import time
try:
    time_clock = time.process_time
except:
    time_clock = time.clock
import testcommon
import backtrader as bt
import backtrader.indicators as btind
BUYCREATE = ['3641.42', '3798.46', '3874.61', '3860.00', '3843.08', '3648.33', '3526.84', '3632.93', '3788.96', '3841.31', '4045.22', '4052.89']
SELLCREATE = ['3763.73', '3811.45', '3823.11', '3821.97', '3837.86', '3604.33', '3562.56', '3772.21', '3780.18', '3974.62', '4048.16']
BUYEXEC = ['3643.35', '3801.03', '3872.37', '3863.57', '3845.32', '3656.43', '3542.65', '3639.65', '3799.86', '3840.20', '4047.63', '4052.55']
SELLEXEC = ['3763.95', '3811.85', '3822.35', '3822.57', '3829.82', '3598.58', '3545.92', '3766.80', '3782.15', '3979.73', '4045.05']

class TestStrategy(bt.Strategy):
    params = (('period', 15), ('printdata', True), ('printops', True), ('stocklike', True))

    def log(self, txt, dt=None, nodate=False):
        if False:
            while True:
                i = 10
        if not nodate:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print('%s, %s' % (dt.isoformat(), txt))
        else:
            print('---------- %s' % txt)

    def notify_order(self, order):
        if False:
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
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
            print('Hello World!')
        tused = time_clock() - self.tstart
        if self.p.printdata:
            self.log('Time used: %s' % str(tused))
            self.log('Final portfolio value: %.2f' % self.broker.getvalue())
            self.log('Final cash value: %.2f' % self.broker.getcash())
            self.log('-------------------------')
            print('buycreate')
            print(self.buycreate)
            print('sellcreate')
            print(self.sellcreate)
            print('buyexec')
            print(self.buyexec)
            print('sellexec')
            print(self.sellexec)
        else:
            if not self.p.stocklike:
                assert '%.2f' % self.broker.getvalue() == '12795.00'
                assert '%.2f' % self.broker.getcash() == '11795.00'
            else:
                assert '%.2f' % self.broker.getvalue() == '10284.10'
                assert '%.2f' % self.broker.getcash() == '6164.16'
            assert self.buycreate == BUYCREATE
            assert self.sellcreate == SELLCREATE
            assert self.buyexec == BUYEXEC
            assert self.sellexec == SELLEXEC

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
        while True:
            i = 10
    for stlike in [False, True]:
        datas = [testcommon.getdata(i) for i in range(chkdatas)]
        testcommon.runtest(datas, TestStrategy, printdata=main, printops=main, stocklike=stlike, plot=main)
if __name__ == '__main__':
    test_run(main=True)