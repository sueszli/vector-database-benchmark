from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

class SMACrossOver(bt.Strategy):
    params = (('stake', 1), ('period', 30))

    def log(self, txt, dt=None):
        if False:
            i = 10
            return i + 15
        ' Logging function fot this strategy'
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if False:
            i = 10
            return i + 15
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (order.executed.price, order.executed.value, order.executed.comm))
            else:
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (order.executed.price, order.executed.value, order.executed.comm))

    def notify_trade(self, trade):
        if False:
            print('Hello World!')
        if trade.isclosed:
            self.log('TRADE PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

    def __init__(self):
        if False:
            return 10
        sma = btind.SMA(self.data, period=self.p.period)
        self.buysell_sig = btind.CrossOver(self.data, sma)

    def next(self):
        if False:
            i = 10
            return i + 15
        if self.buysell_sig > 0:
            self.log('BUY CREATE, %.2f' % self.data.close[0])
            self.buy(size=self.p.stake)
        elif self.position and self.buysell_sig < 0:
            self.log('SELL CREATE, %.2f' % self.data.close[0])
            self.sell(size=self.p.stake)

def runstrategy():
    if False:
        return 10
    args = parse_args()
    cerebro = bt.Cerebro()
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    data = btfeeds.BacktraderCSVData(dataname=args.data, fromdate=fromdate, todate=todate)
    cerebro.adddata(data)
    cerebro.addstrategy(SMACrossOver, period=args.period, stake=args.stake)
    cerebro.broker.setcash(args.cash)
    commtypes = dict(none=None, perc=bt.CommInfoBase.COMM_PERC, fixed=bt.CommInfoBase.COMM_FIXED)
    cerebro.broker.setcommission(commission=args.comm, mult=args.mult, margin=args.margin, percabs=not args.percrel, commtype=commtypes[args.commtype], stocklike=args.stocklike)
    cerebro.run()
    if args.plot:
        cerebro.plot(numfigs=args.numfigs, volume=False)

def parse_args():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Commission schemes', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', '-d', default='../../datas/2006-day-001.txt', help='data to add to the system')
    parser.add_argument('--fromdate', '-f', default='2006-01-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', default='2006-12-31', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--stake', default=1, type=int, help='Stake to apply in each operation')
    parser.add_argument('--period', default=30, type=int, help='Period to apply to the Simple Moving Average')
    parser.add_argument('--cash', default=10000.0, type=float, help='Starting Cash')
    parser.add_argument('--comm', default=2.0, type=float, help='Commission factor for operation, either apercentage or a per stake unit absolute value')
    parser.add_argument('--mult', default=10, type=int, help='Multiplier for operations calculation')
    parser.add_argument('--margin', default=2000.0, type=float, help='Margin for futures-like operations')
    parser.add_argument('--commtype', required=False, default='none', choices=['none', 'perc', 'fixed'], help='Commission - choose none for the old CommissionInfo behavior')
    parser.add_argument('--stocklike', required=False, action='store_true', help='If the operation is for stock-like assets orfuture-like assets')
    parser.add_argument('--percrel', required=False, action='store_true', help='If perc is expressed in relative xx% ratherthan absolute value 0.xx')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the read data')
    parser.add_argument('--numfigs', '-n', default=1, help='Plot using numfigs figures')
    return parser.parse_args()
if __name__ == '__main__':
    runstrategy()