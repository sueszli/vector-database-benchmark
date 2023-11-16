from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
from backtrader.analyzers import SQN

class LongShortStrategy(bt.Strategy):
    """This strategy buys/sells upong the close price crossing
    upwards/downwards a Simple Moving Average.

    It can be a long-only strategy by setting the param "onlylong" to True
    """
    params = dict(period=15, stake=1, printout=False, onlylong=False, csvcross=False)

    def start(self):
        if False:
            i = 10
            return i + 15
        pass

    def stop(self):
        if False:
            while True:
                i = 10
        pass

    def log(self, txt, dt=None):
        if False:
            i = 10
            return i + 15
        if self.p.printout:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.orderid = None
        sma = btind.MovAv.SMA(self.data, period=self.p.period)
        self.signal = btind.CrossOver(self.data.close, sma)
        self.signal.csv = self.p.csvcross

    def next(self):
        if False:
            i = 10
            return i + 15
        if self.orderid:
            return
        if self.signal > 0.0:
            if self.position:
                self.log('CLOSE SHORT , %.2f' % self.data.close[0])
                self.close()
            self.log('BUY CREATE , %.2f' % self.data.close[0])
            self.buy(size=self.p.stake)
        elif self.signal < 0.0:
            if self.position:
                self.log('CLOSE LONG , %.2f' % self.data.close[0])
                self.close()
            if not self.p.onlylong:
                self.log('SELL CREATE , %.2f' % self.data.close[0])
                self.sell(size=self.p.stake)

    def notify_order(self, order):
        if False:
            while True:
                i = 10
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

    def notify_trade(self, trade):
        if False:
            for i in range(10):
                print('nop')
        if trade.isclosed:
            self.log('TRADE PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))
        elif trade.justopened:
            self.log('TRADE OPENED, SIZE %2d' % trade.size)

def runstrategy():
    if False:
        print('Hello World!')
    args = parse_args()
    cerebro = bt.Cerebro()
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    data = btfeeds.BacktraderCSVData(dataname=args.data, fromdate=fromdate, todate=todate)
    cerebro.adddata(data)
    cerebro.addstrategy(LongShortStrategy, period=args.period, onlylong=args.onlylong, csvcross=args.csvcross, stake=args.stake)
    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=args.comm, mult=args.mult, margin=args.margin)
    cerebro.addanalyzer(SQN)
    cerebro.addwriter(bt.WriterFile, csv=args.writercsv, rounding=2)
    cerebro.run()
    if args.plot:
        cerebro.plot(numfigs=args.numfigs, volume=False, zdown=False)

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='MultiData Strategy')
    parser.add_argument('--data', '-d', default='../../datas/2006-day-001.txt', help='data to add to the system')
    parser.add_argument('--fromdate', '-f', default='2006-01-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', default='2006-12-31', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--period', default=15, type=int, help='Period to apply to the Simple Moving Average')
    parser.add_argument('--onlylong', '-ol', action='store_true', help='Do only long operations')
    parser.add_argument('--writercsv', '-wcsv', action='store_true', help='Tell the writer to produce a csv stream')
    parser.add_argument('--csvcross', action='store_true', help='Output the CrossOver signals to CSV')
    parser.add_argument('--cash', default=100000, type=int, help='Starting Cash')
    parser.add_argument('--comm', default=2, type=float, help='Commission for operation')
    parser.add_argument('--mult', default=10, type=int, help='Multiplier for futures')
    parser.add_argument('--margin', default=2000.0, type=float, help='Margin for each future')
    parser.add_argument('--stake', default=1, type=int, help='Stake to apply in each operation')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the read data')
    parser.add_argument('--numfigs', '-n', default=1, help='Plot using numfigs figures')
    return parser.parse_args()
if __name__ == '__main__':
    runstrategy()