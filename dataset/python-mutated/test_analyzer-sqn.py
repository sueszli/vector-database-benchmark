from __future__ import absolute_import, division, print_function, unicode_literals
import time
try:
    time_clock = time.process_time
except:
    time_clock = time.clock
import testcommon
import backtrader as bt
import backtrader.indicators as btind

class TestStrategy(bt.Strategy):
    params = (('period', 15), ('maxtrades', None), ('printdata', True), ('printops', True), ('stocklike', True))

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

    def notify_trade(self, trade):
        if False:
            while True:
                i = 10
        if trade.isclosed:
            self.tradecount += 1

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
            while True:
                i = 10
        self.orderid = None
        self.sma = btind.SMA(self.data, period=self.p.period)
        self.cross = btind.CrossOver(self.data.close, self.sma, plot=True)

    def start(self):
        if False:
            return 10
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
        self.tradecount = 0

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
        if self.p.printdata:
            self.log('Open, High, Low, Close, %.2f, %.2f, %.2f, %.2f, Sma, %f' % (self.data.open[0], self.data.high[0], self.data.low[0], self.data.close[0], self.sma[0]))
            self.log('Close %.2f - Sma %.2f' % (self.data.close[0], self.sma[0]))
        if self.orderid:
            return
        if not self.position.size:
            if self.p.maxtrades is None or self.tradecount < self.p.maxtrades:
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
        print('Hello World!')
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    for maxtrades in [None, 0, 1]:
        cerebros = testcommon.runtest(datas, TestStrategy, printdata=main, stocklike=False, maxtrades=maxtrades, printops=main, plot=main, analyzer=(bt.analyzers.SQN, {}))
        for cerebro in cerebros:
            strat = cerebro.runstrats[0][0]
            analyzer = strat.analyzers[0]
            analysis = analyzer.get_analysis()
            if main:
                print(analysis)
                print(str(analysis.sqn))
            elif maxtrades == 0 or maxtrades == 1:
                assert analysis.sqn == 0
                assert analysis.trades == maxtrades
            else:
                assert str(analysis.sqn)[0:14] == '0.912550316439'
                assert str(analysis.trades) == '11'
if __name__ == '__main__':
    test_run(main=True)