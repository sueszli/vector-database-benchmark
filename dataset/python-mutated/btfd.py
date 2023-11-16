from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt

class ValueUnlever(bt.observers.Value):
    """Extension of regular Value observer to add leveraged view"""
    lines = ('value_lever', 'asset')
    params = (('assetstart', 100000.0), ('lever', True))

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        super(ValueUnlever, self).next()
        if self.p.lever:
            self.lines.value_lever[0] = self._owner.broker._valuelever
        if len(self) == 1:
            self.lines.asset[0] = self.p.assetstart
        else:
            change = self.data[0] / self.data[-1]
            self.lines.asset[0] = change * self.lines.asset[-1]

class St(bt.Strategy):
    params = (('fall', -0.01), ('hold', 2), ('approach', 'highlow'), ('target', 1.0), ('prorder', False), ('prtrade', False), ('prdata', False))

    def __init__(self):
        if False:
            while True:
                i = 10
        if self.p.approach == 'closeclose':
            self.pctdown = self.data.close / self.data.close(-1) - 1.0
        elif self.p.approach == 'openclose':
            self.pctdown = self.data.close / self.data.open - 1.0
        elif self.p.approach == 'highclose':
            self.pctdown = self.data.close / self.data.high - 1.0
        elif self.p.approach == 'highlow':
            self.pctdown = self.data.low / self.data.high - 1.0

    def next(self):
        if False:
            return 10
        if self.position:
            if len(self) == self.barexit:
                self.close()
                if self.p.prdata:
                    print(','.join((str(x) for x in ['DATA', 'CLOSE', self.data.datetime.date().isoformat(), self.data.close[0], float('NaN')])))
        elif self.pctdown <= self.p.fall:
            self.order_target_percent(target=self.p.target)
            self.barexit = len(self) + self.p.hold
            if self.p.prdata:
                print(','.join((str(x) for x in ['DATA', 'OPEN', self.data.datetime.date().isoformat(), self.data.close[0], self.pctdown[0]])))

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        if self.p.prtrade:
            print(','.join(['TRADE', 'Status', 'Date', 'Value', 'PnL', 'Commission']))
        if self.p.prorder:
            print(','.join(['ORDER', 'Type', 'Date', 'Price', 'Size', 'Commission']))
        if self.p.prdata:
            print(','.join(['DATA', 'Action', 'Date', 'Price', 'PctDown']))

    def notify_order(self, order):
        if False:
            while True:
                i = 10
        if order.status in [order.Margin, order.Rejected, order.Canceled]:
            print('ORDER FAILED with status:', order.getstatusname())
        elif order.status == order.Completed:
            if self.p.prorder:
                print(','.join(map(str, ['ORDER', 'BUY' * order.isbuy() or 'SELL', self.data.num2date(order.executed.dt).date().isoformat(), order.executed.price, order.executed.size, order.executed.comm])))

    def notify_trade(self, trade):
        if False:
            return 10
        if not self.p.prtrade:
            return
        if trade.isclosed:
            print(','.join(map(str, ['TRADE', 'CLOSE', self.data.num2date(trade.dtclose).date().isoformat(), trade.value, trade.pnl, trade.commission])))
        elif trade.justopened:
            print(','.join(map(str, ['TRADE', 'OPEN', self.data.num2date(trade.dtopen).date().isoformat(), trade.value, trade.pnl, trade.commission])))

def runstrat(args=None):
    if False:
        print('Hello World!')
    args = parse_args(args)
    cerebro = bt.Cerebro()
    kwargs = dict()
    (dtfmt, tmfmt) = ('%Y-%m-%d', 'T%H:%M:%S')
    for (a, d) in ((getattr(args, x), x) for x in ['fromdate', 'todate']):
        kwargs[d] = datetime.datetime.strptime(a, dtfmt + tmfmt * ('T' in a))
    if not args.offline:
        YahooData = bt.feeds.YahooFinanceData
    else:
        YahooData = bt.feeds.YahooFinanceCSVData
    data = YahooData(dataname=args.data, plot=False, **kwargs)
    cerebro.adddata(data)
    cerebro.broker = bt.brokers.BackBroker(**eval('dict(' + args.broker + ')'))
    cerebro.broker.setcommission(**eval('dict(' + args.comminfo + ')'))
    cerebro.addstrategy(St, **eval('dict(' + args.strat + ')'))
    cerebro.addobserver(ValueUnlever, **eval('dict(' + args.valobserver + ')'))
    cerebro.run(**eval('dict(' + args.cerebro + ')'))
    if args.plot:
        cerebro.plot(**eval('dict(' + args.plot + ')'))

def parse_args(pargs=None):
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=' - '.join(['BTFD', 'http://dark-bid.com/BTFD-only-strategy-that-matters.html', 'https://www.reddit.com/r/algotrading/comments/5jez2b/can_anyone_replicate_this_strategy/']))
    parser.add_argument('--offline', required=False, action='store_true', help='Use offline file with ticker name')
    parser.add_argument('--data', required=False, default='^GSPC', metavar='TICKER', help='Yahoo ticker to download')
    parser.add_argument('--fromdate', required=False, default='1990-01-01', metavar='YYYY-MM-DD[THH:MM:SS]', help='Starting date[time]')
    parser.add_argument('--todate', required=False, default='2016-10-01', metavar='YYYY-MM-DD[THH:MM:SS]', help='Ending date[time]')
    parser.add_argument('--cerebro', required=False, default='stdstats=False', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--broker', required=False, default='cash=100000.0, coc=True', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--valobserver', required=False, default='assetstart=100000.0', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--strat', required=False, default='approach="highlow"', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--comminfo', required=False, default='leverage=2.0', metavar='kwargs', help='kwargs in key=value format')
    parser.add_argument('--plot', required=False, default='', nargs='?', const='volume=False', metavar='kwargs', help='kwargs in key=value format')
    return parser.parse_args(pargs)
if __name__ == '__main__':
    runstrat()