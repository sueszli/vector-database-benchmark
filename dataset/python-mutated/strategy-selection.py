from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import backtrader as bt

class St0(bt.SignalStrategy):

    def __init__(self):
        if False:
            return 10
        (sma1, sma2) = (bt.ind.SMA(period=10), bt.ind.SMA(period=30))
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)

class St1(bt.SignalStrategy):

    def __init__(self):
        if False:
            print('Hello World!')
        sma1 = bt.ind.SMA(period=10)
        crossover = bt.ind.CrossOver(self.data.close, sma1)
        self.signal_add(bt.SIGNAL_LONG, crossover)

class StFetcher(object):
    _STRATS = [St0, St1]

    def __new__(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        idx = kwargs.pop('idx')
        obj = cls._STRATS[idx](*args, **kwargs)
        return obj

def runstrat(pargs=None):
    if False:
        print('Hello World!')
    args = parse_args(pargs)
    cerebro = bt.Cerebro()
    data = bt.feeds.BacktraderCSVData(dataname=args.data)
    cerebro.adddata(data)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.optstrategy(StFetcher, idx=[0, 1])
    results = cerebro.run(maxcpus=args.maxcpus, optreturn=args.optreturn)
    strats = [x[0] for x in results]
    for (i, strat) in enumerate(strats):
        rets = strat.analyzers.returns.get_analysis()
        print('Strat {} Name {}:\n  - analyzer: {}\n'.format(i, strat.__class__.__name__, rets))

def parse_args(pargs=None):
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample for strategy selection')
    parser.add_argument('--data', required=False, default='../../datas/2005-2006-day-001.txt', help='Data to be read in')
    parser.add_argument('--maxcpus', required=False, action='store', default=None, type=int, help='Limit the numer of CPUs to use')
    parser.add_argument('--optreturn', required=False, action='store_true', help='Return reduced/mocked strategy object')
    return parser.parse_args(pargs)
if __name__ == '__main__':
    runstrat()