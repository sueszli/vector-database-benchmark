from __future__ import absolute_import, division, print_function
import argparse
import datetime
import random
import backtrader as bt
import backtrader.feeds as btfeeds
from backtrader.utils.py3 import with_metaclass

class St(bt.Strategy):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.order = None

    def notify_order(self, order):
        if False:
            for i in range(10):
                print('nop')
        curdtstr = self.data.datetime.datetime().strftime('%a %Y-%m-%d')
        if order.status in [order.Completed]:
            dtstr = bt.num2date(order.executed.dt).strftime('%a %Y-%m-%d')
            if order.isbuy():
                print('%s: BUY  EXECUTED, on:' % curdtstr, dtstr)
            else:
                print('%s: SELL EXECUTED, on:' % curdtstr, dtstr)
            self.order = None

    def next(self):
        if False:
            while True:
                i = 10
        dtstr = self.data.datetime.datetime().strftime('%a %Y-%m-%d %H:%M:%S')
        if self.order:
            return
        if not random.randint(0, 5):
            if self.position:
                print('%s: SELL CREATED' % dtstr)
                self.order = self.close(exectype=bt.Order.Close)
            else:
                print('%s: BUY  CREATED' % dtstr)
                self.order = self.buy(exectype=bt.Order.Close)

class SessionEndFiller(with_metaclass(bt.metabase.MetaParams, object)):
    """This data filter simply adds the time given in param ``endtime`` to the
    current data datetime

    It is intended for daily bars which come from sources with no time
    indication and can be used to signal the bar is passed the end of the
    session

    The default value for ``endtime`` is 1 second before midnight 23:59:59
    """
    params = (('endtime', datetime.time(23, 59, 59)),)

    def __call__(self, data):
        if False:
            while True:
                i = 10
        '\n        Params:\n          - data: the data source to filter/process\n\n        Returns:\n          - False (always) because this filter does not remove bars from the\n            stream\n        '
        dtime = datetime.combine(data.datetime.date(), self.p.endtime)
        data.datetime[0] = data.date2num(dtime)
        return False

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
        print('Hello World!')
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
    if args.filltime is not None:
        filltime = datetime.datetime.strptime(args.filltime, '%H:%M:%S').time()
        data.addfilter(SessionEndFiller, endtime=filltime)
    return data

def parse_args():
    if False:
        return 10
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample for Close Orders with daily data')
    parser.add_argument('--infile', '-i', required=False, default='../../datas/2005-2006-day-001.txt', help='File to be read in')
    parser.add_argument('--csvformat', '-c', required=False, default='bt', choices=['bt', 'visualchart', 'sierrachart', 'yahoo', 'yahoo_unreversed'], help='CSV Format')
    parser.add_argument('--fromdate', '-f', required=False, default=None, help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', required=False, default=None, help='Ending date in YYYY-MM-DD format')
    parser.add_argument('--eosbar', required=False, action='store_true', help='Consider a bar with the end of session time tobe the end of the session')
    parser.add_argument('--tend', '-te', default=None, required=False, help='End time for the Session Filter (HH:MM)')
    parser.add_argument('--filltime', '-ftime', default=None, required=False, help='Add Time to daily bars (HH:MM:SS)')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()