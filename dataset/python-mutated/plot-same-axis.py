from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

class PlotStrategy(bt.Strategy):
    """
    The strategy does nothing but create indicators for plotting purposes
    """
    params = dict(smasubplot=False, nomacdplot=False, rsioverstoc=False, rsioversma=False, stocrsi=False, stocrsilabels=False)

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        sma = btind.SMA(subplot=self.params.smasubplot)
        macd = btind.MACD()
        macd.plotinfo.plot = not self.params.nomacdplot
        stoc = btind.Stochastic()
        rsi = btind.RSI()
        if self.params.stocrsi:
            stoc.plotinfo.plotmaster = rsi
            stoc.plotinfo.plotlinelabels = self.p.stocrsilabels
        elif self.params.rsioverstoc:
            rsi.plotinfo.plotmaster = stoc
        elif self.params.rsioversma:
            rsi.plotinfo.plotmaster = sma

def runstrategy():
    if False:
        while True:
            i = 10
    args = parse_args()
    cerebro = bt.Cerebro()
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    data = btfeeds.BacktraderCSVData(dataname=args.data, fromdate=fromdate, todate=todate)
    cerebro.adddata(data)
    cerebro.addstrategy(PlotStrategy, smasubplot=args.smasubplot, nomacdplot=args.nomacdplot, rsioverstoc=args.rsioverstoc, rsioversma=args.rsioversma, stocrsi=args.stocrsi, stocrsilabels=args.stocrsilabels)
    cerebro.run(stdstats=args.stdstats)
    cerebro.plot(numfigs=args.numfigs, volume=False)

def parse_args():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Plotting Example')
    parser.add_argument('--data', '-d', default='../../datas/2006-day-001.txt', help='data to add to the system')
    parser.add_argument('--fromdate', '-f', default='2006-01-01', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', default='2006-12-31', help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--stdstats', '-st', action='store_true', help='Show standard observers')
    parser.add_argument('--smasubplot', '-ss', action='store_true', help='Put SMA on own subplot/axis')
    parser.add_argument('--nomacdplot', '-nm', action='store_true', help='Hide the indicator from the plot')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--rsioverstoc', '-ros', action='store_true', help='Plot the RSI indicator on the Stochastic axis')
    group.add_argument('--rsioversma', '-rom', action='store_true', help='Plot the RSI indicator on the SMA axis')
    group.add_argument('--stocrsi', '-strsi', action='store_true', help='Plot the Stochastic indicator on the RSI axis')
    parser.add_argument('--stocrsilabels', action='store_true', help='Plot line names instead of indicator name')
    parser.add_argument('--numfigs', '-n', default=1, help='Plot using numfigs figures')
    return parser.parse_args()
if __name__ == '__main__':
    runstrategy()