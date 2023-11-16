from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import backtrader as bt

class St(bt.Strategy):
    params = ()

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def start(self):
        if False:
            print('Hello World!')
        self.callcounter = 0
        txtfields = list()
        txtfields.append('Calls')
        txtfields.append('Len Strat')
        txtfields.append('Len Data')
        txtfields.append('Datetime')
        txtfields.append('Open')
        txtfields.append('High')
        txtfields.append('Low')
        txtfields.append('Close')
        txtfields.append('Volume')
        txtfields.append('OpenInterest')
        print(','.join(txtfields))
        self.lcontrol = 0

    def next(self):
        if False:
            i = 10
            return i + 15
        self.callcounter += 1
        txtfields = list()
        txtfields.append('%04d' % self.callcounter)
        txtfields.append('%04d' % len(self))
        txtfields.append('%04d' % len(self.data0))
        txtfields.append(self.data.datetime.datetime(0).isoformat())
        txtfields.append('%.2f' % self.data0.open[0])
        txtfields.append('%.2f' % self.data0.high[0])
        txtfields.append('%.2f' % self.data0.low[0])
        txtfields.append('%.2f' % self.data0.close[0])
        txtfields.append('%.2f' % self.data0.volume[0])
        txtfields.append('%.2f' % self.data0.openinterest[0])
        print(','.join(txtfields))
        if len(self.data) > self.lcontrol:
            print('- I could issue a buy order during the Opening')
        self.lcontrol = len(self.data)

def runstrat():
    if False:
        return 10
    args = parse_args()
    cerebro = bt.Cerebro()
    data = bt.feeds.BacktraderCSVData(dataname=args.data)
    data.addfilter(bt.filters.DayStepsFilter)
    cerebro.adddata(data)
    cerebro.addstrategy(St)
    cerebro._doreplay = True
    cerebro.run(**eval('dict(' + args.cerebro + ')'))
    if args.plot:
        cerebro.plot(**eval('dict(' + args.plot + ')'))

def parse_args(pargs=None):
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample for pivot point and cross plotting')
    parser.add_argument('--data', required=False, default='../../datas/2005-2006-day-001.txt', help='Data to be read in')
    parser.add_argument('--cerebro', required=False, action='store', default='', help='Arguments for cerebro')
    parser.add_argument('--plot', '-p', nargs='?', required=False, metavar='kwargs', const='{}', help='Plot (with additional args if passed')
    return parser.parse_args(pargs)
if __name__ == '__main__':
    runstrat()