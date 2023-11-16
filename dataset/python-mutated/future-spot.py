from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import random
import backtrader as bt

def close_changer(data, *args, **kwargs):
    if False:
        while True:
            i = 10
    data.close[0] += 50.0 * random.randint(-1, 1)
    return False

class BuySellArrows(bt.observers.BuySell):
    plotlines = dict(buy=dict(marker='$⇧$', markersize=12.0), sell=dict(marker='$⇩$', markersize=12.0))

class St(bt.Strategy):

    def __init__(self):
        if False:
            print('Hello World!')
        bt.obs.BuySell(self.data0, barplot=True)
        BuySellArrows(self.data1, barplot=True)

    def next(self):
        if False:
            while True:
                i = 10
        if not self.position:
            if random.randint(0, 1):
                self.buy(data=self.data0)
                self.entered = len(self)
        elif len(self) - self.entered >= 10:
            self.sell(data=self.data1)

def runstrat(args=None):
    if False:
        i = 10
        return i + 15
    args = parse_args(args)
    cerebro = bt.Cerebro()
    dataname = '../../datas/2006-day-001.txt'
    data0 = bt.feeds.BacktraderCSVData(dataname=dataname, name='data0')
    cerebro.adddata(data0)
    data1 = bt.feeds.BacktraderCSVData(dataname=dataname, name='data1')
    data1.addfilter(close_changer)
    if not args.no_comp:
        data1.compensate(data0)
    data1.plotinfo.plotmaster = data0
    if args.sameaxis:
        data1.plotinfo.sameaxis = True
    cerebro.adddata(data1)
    cerebro.addstrategy(St)
    cerebro.addobserver(bt.obs.Broker)
    cerebro.addobserver(bt.obs.Trades)
    cerebro.broker.set_coc(True)
    cerebro.run(stdstats=False)
    cerebro.plot(volume=False)

def parse_args(pargs=None):
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Compensation example')
    parser.add_argument('--no-comp', required=False, action='store_true')
    parser.add_argument('--sameaxis', required=False, action='store_true')
    return parser.parse_args(pargs)
if __name__ == '__main__':
    runstrat()