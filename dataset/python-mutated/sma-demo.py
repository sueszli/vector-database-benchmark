import backtrader as bt
import datetime

class TestStrategy(bt.Strategy):
    """
    继承并构建自己的bt策略
    """

    def log(self, txt, dt=None, doprint=False):
        if False:
            for i in range(10):
                print('nop')
        ' 日志函数，用于统一输出日志格式 '
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        if False:
            print('Hello World!')
        print('数据源长度 ', len(self.datas))
        self.dataclose = self.datas[0].close
        print(len(self.dataclose))
        print(type(self.dataclose))
        print(self.dataclose[0])
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.sma5 = bt.indicators.SimpleMovingAverage(self.datas[0], period=5)
        self.sma10 = bt.indicators.SimpleMovingAverage(self.datas[0], period=10)

    def notify_order(self, order):
        if False:
            for i in range(10):
                print('nop')
        '\n        订单状态处理\n\n        Arguments:\n            order {object} -- 订单状态\n        '
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def notify_trade(self, trade):
        if False:
            while True:
                i = 10
        '\n        交易成果\n        \n        Arguments:\n            trade {object} -- 交易状态\n        '
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm), doprint=True)

    def next(self):
        if False:
            while True:
                i = 10
        ' 下一次执行 '
        self.log('Close, %.2f' % self.dataclose[0])
        if self.order:
            return
        if not self.position:
            if self.sma5[0] > self.sma10[0]:
                self.order = self.buy()
        elif self.sma5[0] < self.sma10[0]:
            self.order = self.sell()

    def stop(self):
        if False:
            print('Hello World!')
        self.log(u'(回测结束) Ending Value %.2f' % self.broker.getvalue(), doprint=True)
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    strats = cerebro.addstrategy(TestStrategy)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)
    data = bt.feeds.GenericCSVData(dataname='600519.csv', fromdate=datetime.datetime(2010, 1, 1), todate=datetime.datetime(2020, 4, 12), dtformat='%Y%m%d', datetime=2, open=3, high=4, low=5, close=6, volume=10)
    cerebro.adddata(data)
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.setcommission(0.005)
    print('启动资金: %.2f' % cerebro.broker.getvalue())
    cerebro.run()