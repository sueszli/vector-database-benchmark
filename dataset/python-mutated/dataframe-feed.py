from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import backtrader as bt
import backtrader.feeds as btfeeds
from datapath import ROOT
import pandas
import datetime

def convert_time(x):
    if False:
        i = 10
        return i + 15
    return datetime.datetime.fromtimestamp(int(x / 1000))

def runstrat():
    if False:
        return 10
    args = parse_args()
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(bt.Strategy)
    ROOT = '/home/xda/github/stock_strategy/backtrader-code/'
    datapath = ROOT + 'BNBUSDT.csv'
    skiprows = 1 if args.noheaders else 0
    header = None if args.noheaders else 0
    dataframe = pandas.read_csv(datapath, skiprows=skiprows, header=header)
    dataframe['open_time'] = dataframe['open_time'].map(convert_time)
    dataframe = dataframe.set_index('open_time', drop=True)
    if not args.noprint:
        print('--------------------------------------------------')
        print(dataframe)
        print('--------------------------------------------------')
    data = bt.feeds.PandasData(dataname=dataframe)
    cerebro.adddata(data)
    cerebro.run()
    cerebro.plot(style='bar')

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Pandas test script')
    parser.add_argument('--noheaders', action='store_true', default=False, required=False, help='Do not use header rows')
    parser.add_argument('--noprint', action='store_true', default=False, help='Print the dataframe')
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()