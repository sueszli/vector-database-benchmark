from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

class PairTradingStrategy(bt.Strategy):
    params = dict(period=10, stake=10, qty1=0, qty2=0, printout=True, upper=2.1, lower=-2.1, up_medium=0.5, low_medium=-0.5, status=0, portfolio_value=10000)

    def log(self, txt, dt=None):
        if False:
            while True:
                i = 10
        if self.p.printout:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if False:
            for i in range(10):
                print('nop')
        if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                buytxt = 'BUY COMPLETE, %.2f' % order.executed.price
                self.log(buytxt, order.executed.dt)
            else:
                selltxt = 'SELL COMPLETE, %.2f' % order.executed.price
                self.log(selltxt, order.executed.dt)
        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            self.log('%s ,' % order.Status[order.status])
            pass
        self.orderid = None

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.orderid = None
        self.qty1 = self.p.qty1
        self.qty2 = self.p.qty2
        self.upper_limit = self.p.upper
        self.lower_limit = self.p.lower
        self.up_medium = self.p.up_medium
        self.low_medium = self.p.low_medium
        self.status = self.p.status
        self.portfolio_value = self.p.portfolio_value
        self.transform = btind.OLS_TransformationN(self.data0, self.data1, period=self.p.period)
        self.zscore = self.transform.zscore

    def next(self):
        if False:
            while True:
                i = 10
        if self.orderid:
            return
        if self.p.printout:
            print('Self  len:', len(self))
            print('Data0 len:', len(self.data0))
            print('Data1 len:', len(self.data1))
            print('Data0 len == Data1 len:', len(self.data0) == len(self.data1))
            print('Data0 dt:', self.data0.datetime.datetime())
            print('Data1 dt:', self.data1.datetime.datetime())
        print('status is', self.status)
        print('zscore is', self.zscore[0])
        if self.zscore[0] > self.upper_limit and self.status != 1:
            value = 0.5 * self.portfolio_value
            x = int(value / self.data0.close)
            y = int(value / self.data1.close)
            print('x + self.qty1 is', x + self.qty1)
            print('y + self.qty2 is', y + self.qty2)
            self.log('SELL CREATE %s, price = %.2f, qty = %d' % ('PEP', self.data0.close[0], x + self.qty1))
            self.sell(data=self.data0, size=x + self.qty1)
            self.log('BUY CREATE %s, price = %.2f, qty = %d' % ('KO', self.data1.close[0], y + self.qty2))
            self.buy(data=self.data1, size=y + self.qty2)
            self.qty1 = x
            self.qty2 = y
            self.status = 1
        elif self.zscore[0] < self.lower_limit and self.status != 2:
            value = 0.5 * self.portfolio_value
            x = int(value / self.data0.close)
            y = int(value / self.data1.close)
            print('x + self.qty1 is', x + self.qty1)
            print('y + self.qty2 is', y + self.qty2)
            self.log('BUY CREATE %s, price = %.2f, qty = %d' % ('PEP', self.data0.close[0], x + self.qty1))
            self.buy(data=self.data0, size=x + self.qty1)
            self.log('SELL CREATE %s, price = %.2f, qty = %d' % ('KO', self.data1.close[0], y + self.qty2))
            self.sell(data=self.data1, size=y + self.qty2)
            self.qty1 = x
            self.qty2 = y
            self.status = 2
        '\n        elif (self.zscore[0] < self.up_medium and self.zscore[0] > self.low_medium):\n            self.log(\'CLOSE LONG %s, price = %.2f\' % ("PEP", self.data0.close[0]))\n            self.close(self.data0)\n            self.log(\'CLOSE LONG %s, price = %.2f\' % ("KO", self.data1.close[0]))\n            self.close(self.data1)\n        '

    def stop(self):
        if False:
            return 10
        print('==================================================')
        print('Starting Value - %.2f' % self.broker.startingcash)
        print('Ending   Value - %.2f' % self.broker.getvalue())
        print('==================================================')

def runstrategy():
    if False:
        for i in range(10):
            print('nop')
    args = parse_args()
    cerebro = bt.Cerebro()
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    data0 = btfeeds.YahooFinanceCSVData(dataname=args.data0, fromdate=fromdate, todate=todate)
    cerebro.adddata(data0)
    data1 = btfeeds.YahooFinanceCSVData(dataname=args.data1, fromdate=fromdate, todate=todate)
    cerebro.adddata(data1)
    cerebro.addstrategy(PairTradingStrategy, period=args.period, stake=args.stake)
    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=args.commperc)
    cerebro.run(runonce=not args.runnext, preload=not args.nopreload, oldsync=args.oldsync)
    if args.plot:
        cerebro.plot(numfigs=args.numfigs, volume=False, zdown=False)

def parse_args():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='MultiData Strategy')
    parser.add_argument('--data0', '-d0', default='../../datas/daily-PEP.csv', help='1st data into the system')
    parser.add_argument('--data1', '-d1', default='../../datas/daily-KO.csv', help='2nd data into the system')
    parser.add_argument('--fromdate', '-f', default='1997-01-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', default='1998-06-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--period', default=10, type=int, help='Period to apply to the Simple Moving Average')
    parser.add_argument('--cash', default=100000, type=int, help='Starting Cash')
    parser.add_argument('--runnext', action='store_true', help='Use next by next instead of runonce')
    parser.add_argument('--nopreload', action='store_true', help='Do not preload the data')
    parser.add_argument('--oldsync', action='store_true', help='Use old data synchronization method')
    parser.add_argument('--commperc', default=0.005, type=float, help='Percentage commission (0.005 is 0.5%%')
    parser.add_argument('--stake', default=10, type=int, help='Stake to apply in each operation')
    parser.add_argument('--plot', '-p', default=True, action='store_true', help='Plot the read data')
    parser.add_argument('--numfigs', '-n', default=1, help='Plot using numfigs figures')
    return parser.parse_args()
if __name__ == '__main__':
    runstrategy()