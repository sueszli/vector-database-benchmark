from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import backtrader as bt
import backtrader.feeds as btfeeds
import pandas

class PandasDataOptix(btfeeds.PandasData):
    lines = ('optix_close', 'optix_pess', 'optix_opt')
    params = (('optix_close', -1), ('optix_pess', -1), ('optix_opt', -1))
    if False:
        datafields = btfeeds.PandasData.datafields + ['optix_close', 'optix_pess', 'optix_opt']

class StrategyOptix(bt.Strategy):

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        print('%03d %f %f, %f' % (len(self), self.data.optix_close[0], self.data.lines.optix_pess[0], self.data.optix_opt[0]))

def runstrat():
    if False:
        return 10
    args = parse_args()
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(StrategyOptix)
    datapath = '../../datas/2006-day-001-optix.txt'
    skiprows = 1 if args.noheaders else 0
    header = None if args.noheaders else 0
    dataframe = pandas.read_csv(datapath, skiprows=skiprows, header=header, parse_dates=True, index_col=0)
    if not args.noprint:
        print('--------------------------------------------------')
        print(dataframe)
        print('--------------------------------------------------')
    data = PandasDataOptix(dataname=dataframe)
    cerebro.adddata(data)
    cerebro.run()
    if not args.noplot:
        cerebro.plot(style='bar')

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Pandas test script')
    parser.add_argument('--noheaders', action='store_true', default=False, required=False, help='Do not use header rows')
    parser.add_argument('--noprint', action='store_true', default=False, help='Print the dataframe')
    parser.add_argument('--noplot', action='store_true', default=False, help='Do not plot the chart')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()