from __future__ import absolute_import, division, print_function
import argparse
import datetime
import backtrader as bt
import backtrader.feeds as btfeeds

class St(bt.Strategy):

    def __init__(self):
        if False:
            return 10
        self.curdate = datetime.date.min
        self.elapsed = 0
        self.order = None

    def notify_order(self, order):
        if False:
            for i in range(10):
                print('nop')
        curdtstr = self.data.datetime.datetime().strftime('%a %Y-%m-%d %H:%M:%S')
        if order.status in [order.Completed]:
            dtstr = bt.num2date(order.executed.dt).strftime('%a %Y-%m-%d %H:%M:%S')
            if order.isbuy():
                print('%s: BUY  EXECUTED, on:' % curdtstr, dtstr)
                self.order = None
            else:
                print('%s: SELL EXECUTED, on:' % curdtstr, dtstr)

    def next(self):
        if False:
            print('Hello World!')
        curdate = self.data.datetime.date()
        if curdate > self.curdate:
            self.elapsed += 1
            self.curdate = curdate
        dtstr = self.data.datetime.datetime().strftime('%a %Y-%m-%d %H:%M:%S')
        if self.position and self.elapsed == 2:
            print('%s: SELL CREATED' % dtstr)
            self.close(exectype=bt.Order.Close)
            self.elapsed = 0
        elif self.order is None and self.elapsed == 2:
            print('%s: BUY  CREATED' % dtstr)
            self.order = self.buy(exectype=bt.Order.Close)
            self.elapsed = 0

def runstrat():
    if False:
        for i in range(10):
            print('nop')
    args = parse_args()
    cerebro = bt.Cerebro()
    cerebro.adddata(getdata(args))
    cerebro.addstrategy(St)
    if args.eosbar:
        cerebro.broker.seteosbar(True)
    cerebro.run()

def getdata(args):
    if False:
        for i in range(10):
            print('nop')
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
    if args.tend is not None:
        dfkwargs['sessionend'] = datetime.datetime.strptime(args.tend, '%H:%M')
    dfkwargs['dataname'] = args.infile
    dfcls = dataformat[args.csvformat]
    data = dfcls(**dfkwargs)
    return data

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample for Close Orders with daily data')
    parser.add_argument('--infile', '-i', required=False, default='../../datas/2006-min-005.txt', help='File to be read in')
    parser.add_argument('--csvformat', '-c', required=False, default='bt', choices=['bt', 'visualchart', 'sierrachart', 'yahoo', 'yahoo_unreversed'], help='CSV Format')
    parser.add_argument('--fromdate', '-f', required=False, default=None, help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', required=False, default=None, help='Ending date in YYYY-MM-DD format')
    parser.add_argument('--eosbar', required=False, action='store_true', help='Consider a bar with the end of session time tobe the end of the session')
    parser.add_argument('--tend', '-te', default=None, required=False, help='End time for the Session Filter (HH:MM)')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()