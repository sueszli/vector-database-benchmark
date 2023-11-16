from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import backtrader as bt
import backtrader.feeds as btfeeds
import pandas

def runstrat():
    if False:
        for i in range(10):
            print('nop')
    args = parse_args()
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(bt.Strategy)
    datapath = '../../datas/2006-day-001.txt'
    skiprows = 1 if args.noheaders else 0
    header = None if args.noheaders else 0
    dataframe = pandas.read_csv(datapath, skiprows=skiprows, header=header, parse_dates=True, index_col=0)
    if not args.noprint:
        print('--------------------------------------------------')
        print(dataframe)
        print('--------------------------------------------------')
    data = bt.feeds.PandasData(dataname=dataframe, nocase=True)
    cerebro.adddata(data)
    cerebro.run()
    cerebro.plot(style='bar')

def parse_args():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Pandas test script')
    parser.add_argument('--noheaders', action='store_true', default=False, required=False, help='Do not use header rows')
    parser.add_argument('--noprint', action='store_true', default=False, help='Print the dataframe')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()