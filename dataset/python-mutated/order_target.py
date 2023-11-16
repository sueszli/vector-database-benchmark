from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from datetime import datetime
import backtrader as bt

class TheStrategy(bt.Strategy):
    """
    This strategy is loosely based on some of the examples from the Van
    K. Tharp book: *Trade Your Way To Financial Freedom*. The logic:

      - Enter the market if:
        - The MACD.macd line crosses the MACD.signal line to the upside
        - The Simple Moving Average has a negative direction in the last x
          periods (actual value below value x periods ago)

     - Set a stop price x times the ATR value away from the close

     - If in the market:

       - Check if the current close has gone below the stop price. If yes,
         exit.
       - If not, update the stop price if the new stop price would be higher
         than the current
    """
    params = (('use_target_size', False), ('use_target_value', False), ('use_target_percent', False))

    def notify_order(self, order):
        if False:
            i = 10
            return i + 15
        if order.status == order.Completed:
            pass
        if not order.alive():
            self.order = None

    def start(self):
        if False:
            print('Hello World!')
        self.order = None

    def next(self):
        if False:
            i = 10
            return i + 15
        dt = self.data.datetime.date()
        portfolio_value = self.broker.get_value()
        print('%04d - %s - Position Size:     %02d - Value %.2f' % (len(self), dt.isoformat(), self.position.size, portfolio_value))
        data_value = self.broker.get_value([self.data])
        if self.p.use_target_value:
            print('%04d - %s - data value %.2f' % (len(self), dt.isoformat(), data_value))
        elif self.p.use_target_percent:
            port_perc = data_value / portfolio_value
            print('%04d - %s - data percent %.2f' % (len(self), dt.isoformat(), port_perc))
        if self.order:
            return
        size = dt.day
        if dt.month % 2 == 0:
            size = 31 - size
        if self.p.use_target_size:
            target = size
            print('%04d - %s - Order Target Size: %02d' % (len(self), dt.isoformat(), size))
            self.order = self.order_target_size(target=size)
        elif self.p.use_target_value:
            value = size * 1000
            print('%04d - %s - Order Target Value: %.2f' % (len(self), dt.isoformat(), value))
            self.order = self.order_target_value(target=value)
        elif self.p.use_target_percent:
            percent = size / 100.0
            print('%04d - %s - Order Target Percent: %.2f' % (len(self), dt.isoformat(), percent))
            self.order = self.order_target_percent(target=percent)

def runstrat(args=None):
    if False:
        print('Hello World!')
    args = parse_args(args)
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.cash)
    dkwargs = dict()
    if args.fromdate is not None:
        dkwargs['fromdate'] = datetime.strptime(args.fromdate, '%Y-%m-%d')
    if args.todate is not None:
        dkwargs['todate'] = datetime.strptime(args.todate, '%Y-%m-%d')
    data = bt.feeds.YahooFinanceCSVData(dataname=args.data, **dkwargs)
    cerebro.adddata(data)
    cerebro.addstrategy(TheStrategy, use_target_size=args.target_size, use_target_value=args.target_value, use_target_percent=args.target_percent)
    cerebro.run()
    if args.plot:
        pkwargs = dict(style='bar')
        if args.plot is not True:
            npkwargs = eval('dict(' + args.plot + ')')
            pkwargs.update(npkwargs)
        cerebro.plot(**pkwargs)

def parse_args(pargs=None):
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample for Order Target')
    parser.add_argument('--data', required=False, default='../../datas/yhoo-1996-2015.txt', help='Specific data to be read in')
    parser.add_argument('--fromdate', required=False, default='2005-01-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', required=False, default='2006-12-31', help='Ending date in YYYY-MM-DD format')
    parser.add_argument('--cash', required=False, action='store', type=float, default=1000000, help='Ending date in YYYY-MM-DD format')
    pgroup = parser.add_mutually_exclusive_group(required=True)
    pgroup.add_argument('--target-size', required=False, action='store_true', help='Use order_target_size')
    pgroup.add_argument('--target-value', required=False, action='store_true', help='Use order_target_value')
    pgroup.add_argument('--target-percent', required=False, action='store_true', help='Use order_target_percent')
    parser.add_argument('--plot', '-p', nargs='?', required=False, metavar='kwargs', const=True, help='Plot the read data applying any kwargs passed\n\nFor example:\n\n  --plot style="candle" (to plot candles)\n')
    if pargs is not None:
        return parser.parse_args(pargs)
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()