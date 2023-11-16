from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import os.path
import time
import sys
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

class MyStrategy(bt.Strategy):
    params = (('smaperiod', 15),)

    def log(self, txt, dt=None):
        if False:
            while True:
                i = 10
        ' Logging function fot this strategy'
        dt = dt or self.data.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        if False:
            while True:
                i = 10
        sma = btind.SMA(period=self.p.smaperiod)
        self.buysell = btind.CrossOver(self.data.close, sma, plot=True)
        self.order = None

    def next(self):
        if False:
            return 10
        self.log('DrawDown: %.2f' % self.stats.drawdown.drawdown[-1])
        self.log('MaxDrawDown: %.2f' % self.stats.drawdown.maxdrawdown[-1])
        if self.position:
            if self.buysell < 0:
                self.log('SELL CREATE, %.2f' % self.data.close[0])
                self.sell()
        elif self.buysell > 0:
            self.log('BUY CREATE, %.2f' % self.data.close[0])
            self.buy()

def runstrat():
    if False:
        i = 10
        return i + 15
    cerebro = bt.Cerebro()
    data = bt.feeds.BacktraderCSVData(dataname='../../datas/2006-day-001.txt')
    cerebro.adddata(data)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.DrawDown_Old)
    cerebro.addstrategy(MyStrategy)
    cerebro.run()
    cerebro.plot()
if __name__ == '__main__':
    runstrat()