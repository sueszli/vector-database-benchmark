from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import os.path
import time
import sys
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

class OrderExecutionStrategy(bt.Strategy):
    params = (('smaperiod', 15), ('exectype', 'Market'), ('perc1', 3), ('perc2', 1), ('valid', 4))

    def log(self, txt, dt=None):
        if False:
            return 10
        ' Logging function fot this strategy'
        dt = dt or self.data.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if False:
            i = 10
            return i + 15
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
            while True:
                i = 10
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
            if self.p.valid:
                valid = self.data.datetime.date(0) + datetime.timedelta(days=self.p.valid)
            else:
                valid = None
            if self.p.exectype == 'Market':
                self.buy(exectype=bt.Order.Market)
                self.log('BUY CREATE, exectype Market, price %.2f' % self.data.close[0])
            elif self.p.exectype == 'Close':
                self.buy(exectype=bt.Order.Close)
                self.log('BUY CREATE, exectype Close, price %.2f' % self.data.close[0])
            elif self.p.exectype == 'Limit':
                price = self.data.close * (1.0 - self.p.perc1 / 100.0)
                self.buy(exectype=bt.Order.Limit, price=price, valid=valid)
                if self.p.valid:
                    txt = 'BUY CREATE, exectype Limit, price %.2f, valid: %s'
                    self.log(txt % (price, valid.strftime('%Y-%m-%d')))
                else:
                    txt = 'BUY CREATE, exectype Limit, price %.2f'
                    self.log(txt % price)
            elif self.p.exectype == 'Stop':
                price = self.data.close * (1.0 + self.p.perc1 / 100.0)
                self.buy(exectype=bt.Order.Stop, price=price, valid=valid)
                if self.p.valid:
                    txt = 'BUY CREATE, exectype Stop, price %.2f, valid: %s'
                    self.log(txt % (price, valid.strftime('%Y-%m-%d')))
                else:
                    txt = 'BUY CREATE, exectype Stop, price %.2f'
                    self.log(txt % price)
            elif self.p.exectype == 'StopLimit':
                price = self.data.close * (1.0 + self.p.perc1 / 100.0)
                plimit = self.data.close * (1.0 + self.p.perc2 / 100.0)
                self.buy(exectype=bt.Order.StopLimit, price=price, valid=valid, plimit=plimit)
                if self.p.valid:
                    txt = 'BUY CREATE, exectype StopLimit, price %.2f, valid: %s, pricelimit: %.2f'
                    self.log(txt % (price, valid.strftime('%Y-%m-%d'), plimit))
                else:
                    txt = 'BUY CREATE, exectype StopLimit, price %.2f, pricelimit: %.2f'
                    self.log(txt % (price, plimit))

def runstrat():
    if False:
        print('Hello World!')
    args = parse_args()
    cerebro = bt.Cerebro()
    data = getdata(args)
    cerebro.adddata(data)
    cerebro.addstrategy(OrderExecutionStrategy, exectype=args.exectype, perc1=args.perc1, perc2=args.perc2, valid=args.valid, smaperiod=args.smaperiod)
    cerebro.run()
    if args.plot:
        cerebro.plot(numfigs=args.numfigs, style=args.plotstyle)

def getdata(args):
    if False:
        while True:
            i = 10
    dataformat = dict(bt=btfeeds.BacktraderCSVData, visualchart=btfeeds.VChartCSVData, sierrachart=btfeeds.SierraChartCSVData, yahoo=btfeeds.YahooFinanceCSVData, yahoo_unreversed=btfeeds.YahooFinanceCSVData)
    dfkwargs = dict()
    if args.csvformat == 'yahoo_unreversed':
        dfkwargs['reverse'] = True
    if args.fromdate:
        fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
        dfkwargs['fromdate'] = fromdate
    if args.todate:
        fromdate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
        dfkwargs['todate'] = todate
    dfkwargs['dataname'] = args.infile
    dfcls = dataformat[args.csvformat]
    return dfcls(**dfkwargs)

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Showcase for Order Execution Types')
    parser.add_argument('--infile', '-i', required=False, default='../../datas/2006-day-001.txt', help='File to be read in')
    parser.add_argument('--csvformat', '-c', required=False, default='bt', choices=['bt', 'visualchart', 'sierrachart', 'yahoo', 'yahoo_unreversed'], help='CSV Format')
    parser.add_argument('--fromdate', '-f', required=False, default=None, help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', required=False, default=None, help='Ending date in YYYY-MM-DD format')
    parser.add_argument('--plot', '-p', action='store_true', required=False, help='Plot the read data')
    parser.add_argument('--plotstyle', '-ps', required=False, default='bar', choices=['bar', 'line', 'candle'], help='Plot the read data')
    parser.add_argument('--numfigs', '-n', required=False, default=1, help='Plot using n figures')
    parser.add_argument('--smaperiod', '-s', required=False, default=15, help='Simple Moving Average Period')
    parser.add_argument('--exectype', '-e', required=False, default='Market', help='Execution Type: Market (default), Close, Limit, Stop, StopLimit')
    parser.add_argument('--valid', '-v', required=False, default=0, type=int, help='Validity for Limit sample: default 0 days')
    parser.add_argument('--perc1', '-p1', required=False, default=0.0, type=float, help='%% distance from close price at order creation time for the limit/trigger price in Limit/Stop orders')
    parser.add_argument('--perc2', '-p2', required=False, default=0.0, type=float, help='%% distance from close price at order creation time for the limit price in StopLimit orders')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()