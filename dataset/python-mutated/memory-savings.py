from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import sys
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
import backtrader.utils.flushfile

class TestInd(bt.Indicator):
    lines = ('a', 'b')

    def __init__(self):
        if False:
            while True:
                i = 10
        self.lines.a = b = self.data.close - self.data.high
        self.lines.b = btind.SMA(b, period=20)

class St(bt.Strategy):
    params = (('datalines', False), ('lendetails', False))

    def __init__(self):
        if False:
            return 10
        btind.SMA()
        btind.Stochastic()
        btind.RSI()
        btind.MACD()
        btind.CCI()
        TestInd().plotinfo.plot = False

    def next(self):
        if False:
            i = 10
            return i + 15
        if self.p.datalines:
            txt = ','.join(['%04d' % len(self), '%04d' % len(self.data0), self.data.datetime.date(0).isoformat()])
            print(txt)

    def loglendetails(self, msg):
        if False:
            i = 10
            return i + 15
        if self.p.lendetails:
            print(msg)

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        super(St, self).stop()
        tlen = 0
        self.loglendetails('-- Evaluating Datas')
        for (i, data) in enumerate(self.datas):
            tdata = 0
            for line in data.lines:
                tdata += len(line.array)
                tline = len(line.array)
            tlen += tdata
            logtxt = '---- Data {} Total Cells {} - Cells per Line {}'
            self.loglendetails(logtxt.format(i, tdata, tline))
        self.loglendetails('-- Evaluating Indicators')
        for (i, ind) in enumerate(self.getindicators()):
            tlen += self.rindicator(ind, i, 0)
        self.loglendetails('-- Evaluating Observers')
        for (i, obs) in enumerate(self.getobservers()):
            tobs = 0
            for line in obs.lines:
                tobs += len(line.array)
                tline = len(line.array)
            tlen += tdata
            logtxt = '---- Observer {} Total Cells {} - Cells per Line {}'
            self.loglendetails(logtxt.format(i, tobs, tline))
        print('Total memory cells used: {}'.format(tlen))

    def rindicator(self, ind, i, deep):
        if False:
            for i in range(10):
                print('nop')
        tind = 0
        for line in ind.lines:
            tind += len(line.array)
            tline = len(line.array)
        thisind = tind
        tsub = 0
        for (j, sind) in enumerate(ind.getindicators()):
            tsub += self.rindicator(sind, j, deep + 1)
        iname = ind.__class__.__name__.split('.')[-1]
        logtxt = '---- Indicator {}.{} {} Total Cells {} - Cells per line {}'
        self.loglendetails(logtxt.format(deep, i, iname, tind, tline))
        logtxt = '---- SubIndicators Total Cells {}'
        self.loglendetails(logtxt.format(deep, i, iname, tsub))
        return tind + tsub

def runstrat():
    if False:
        i = 10
        return i + 15
    args = parse_args()
    cerebro = bt.Cerebro()
    data = btfeeds.YahooFinanceCSVData(dataname=args.data)
    cerebro.adddata(data)
    cerebro.addstrategy(St, datalines=args.datalines, lendetails=args.lendetails)
    cerebro.run(runonce=False, exactbars=args.save)
    if args.plot:
        cerebro.plot(style='bar')

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Check Memory Savings')
    parser.add_argument('--data', required=False, default='../../datas/yhoo-1996-2015.txt', help='Data to be read in')
    parser.add_argument('--save', required=False, type=int, default=0, help='Memory saving level [1, 0, -1, -2]')
    parser.add_argument('--datalines', required=False, action='store_true', help='Print data lines')
    parser.add_argument('--lendetails', required=False, action='store_true', help='Print individual items memory usage')
    parser.add_argument('--plot', required=False, action='store_true', help='Plot the result')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()