from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

class MultiDataStrategy(bt.Strategy):
    """
    This strategy operates on 2 datas. The expectation is that the 2 datas are
    correlated and the 2nd data is used to generate signals on the 1st

      - Buy/Sell Operationss will be executed on the 1st data
      - The signals are generated using a Simple Moving Average on the 2nd data
        when the close price crosses upwwards/downwards

    The strategy is a long-only strategy
    """
    params = dict(period=15, stake=10, printout=True)

    def log(self, txt, dt=None):
        if False:
            return 10
        if self.p.printout:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if False:
            print('Hello World!')
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
            return 10
        self.orderid = None
        sma = btind.MovAv.SMA(self.data1, period=self.p.period)
        self.signal = btind.CrossOver(self.data1.close, sma)

    def next(self):
        if False:
            return 10
        if self.orderid:
            return
        if self.p.printout:
            print('Self  len:', len(self))
            print('Data0 len:', len(self.data0))
            print('Data1 len:', len(self.data1))
            print('Data0 len == Data1 len:', len(self.data0) == len(self.data1))
            print('Data0 dt:', self.data0.datetime.datetime())
            print('Data1 dt:', self.data1.datetime.datetime())
        if not self.position:
            if self.signal > 0.0:
                self.log('BUY CREATE , %.2f' % self.data1.close[0])
                self.buy(size=self.p.stake)
                self.buy(data=self.data1, size=self.p.stake)
        elif self.signal < 0.0:
            self.log('SELL CREATE , %.2f' % self.data1.close[0])
            self.sell(size=self.p.stake)
            self.sell(data=self.data1, size=self.p.stake)

    def stop(self):
        if False:
            return 10
        print('==================================================')
        print('Starting Value - %.2f' % self.broker.startingcash)
        print('Ending   Value - %.2f' % self.broker.getvalue())
        print('==================================================')

def runstrategy():
    if False:
        return 10
    args = parse_args()
    cerebro = bt.Cerebro()
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    data0 = btfeeds.YahooFinanceCSVData(dataname=args.data0, fromdate=fromdate, todate=todate)
    cerebro.adddata(data0)
    data1 = btfeeds.YahooFinanceCSVData(dataname=args.data1, fromdate=fromdate, todate=todate)
    cerebro.adddata(data1)
    cerebro.addstrategy(MultiDataStrategy, period=args.period, stake=args.stake)
    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=args.commperc)
    cerebro.run(runonce=not args.runnext, preload=not args.nopreload, oldsync=args.oldsync)
    if args.plot:
        cerebro.plot(numfigs=args.numfigs, volume=False, zdown=False)

def parse_args():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='MultiData Strategy')
    parser.add_argument('--data0', '-d0', default='../../datas/orcl-1995-2014.txt', help='1st data into the system')
    parser.add_argument('--data1', '-d1', default='../../datas/yhoo-1996-2014.txt', help='2nd data into the system')
    parser.add_argument('--fromdate', '-f', default='2003-01-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', default='2005-12-31', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--period', default=15, type=int, help='Period to apply to the Simple Moving Average')
    parser.add_argument('--cash', default=100000, type=int, help='Starting Cash')
    parser.add_argument('--runnext', action='store_true', help='Use next by next instead of runonce')
    parser.add_argument('--nopreload', action='store_true', help='Do not preload the data')
    parser.add_argument('--oldsync', action='store_true', help='Use old data synchronization method')
    parser.add_argument('--commperc', default=0.005, type=float, help='Percentage commission (0.005 is 0.5%%')
    parser.add_argument('--stake', default=10, type=int, help='Stake to apply in each operation')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the read data')
    parser.add_argument('--numfigs', '-n', default=1, help='Plot using numfigs figures')
    return parser.parse_args()
if __name__ == '__main__':
    runstrategy()