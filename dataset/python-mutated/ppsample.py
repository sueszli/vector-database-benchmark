from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.utils.flushfile

class St(bt.Strategy):
    params = (('usepp1', False), ('plot_on_daily', False))

    def __init__(self):
        if False:
            while True:
                i = 10
        autoplot = self.p.plot_on_daily
        self.pp = pp = bt.ind.PivotPoint(self.data1, _autoplot=autoplot)

    def next(self):
        if False:
            return 10
        txt = ','.join(['%04d' % len(self), '%04d' % len(self.data0), '%04d' % len(self.data1), self.data.datetime.date(0).isoformat(), '%04d' % len(self.pp), '%.2f' % self.pp[0]])
        print(txt)

def runstrat():
    if False:
        while True:
            i = 10
    args = parse_args()
    cerebro = bt.Cerebro()
    data = btfeeds.BacktraderCSVData(dataname=args.data)
    cerebro.adddata(data)
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Months)
    cerebro.addstrategy(St, usepp1=args.usepp1, plot_on_daily=args.plot_on_daily)
    cerebro.run(runonce=False)
    if args.plot:
        cerebro.plot(style='bar')

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample for pivot point and cross plotting')
    parser.add_argument('--data', required=False, default='../../datas/2005-2006-day-001.txt', help='Data to be read in')
    parser.add_argument('--plot', required=False, action='store_true', help='Plot the result')
    parser.add_argument('--plot-on-daily', required=False, action='store_true', help='Plot the indicator on the daily data')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()