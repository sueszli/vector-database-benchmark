from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import backtrader as bt
TFRAMES = dict(days=bt.TimeFrame.Days, weeks=bt.TimeFrame.Weeks, months=bt.TimeFrame.Months, years=bt.TimeFrame.Years)

def runstrat(pargs=None):
    if False:
        for i in range(10):
            print('nop')
    args = parse_args(pargs)
    cerebro = bt.Cerebro()
    if args.cash is not None:
        cerebro.broker.set_cash(args.cash)
    dkwargs = dict()
    if args.fromdate is not None:
        fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
        dkwargs['fromdate'] = fromdate
    if args.todate is not None:
        todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
        dkwargs['todate'] = todate
    data = bt.feeds.BacktraderCSVData(dataname=args.data, **dkwargs)
    cerebro.adddata(data)
    cerebro.addstrategy(bt.strategies.SMA_CrossOver)
    lrkwargs = dict()
    if args.tframe is not None:
        lrkwargs['timeframe'] = TFRAMES[args.tframe]
    if args.tann is not None:
        lrkwargs['tann'] = args.tann
    cerebro.addanalyzer(bt.analyzers.Returns, **lrkwargs)
    vwrkwargs = dict()
    if args.tframe is not None:
        vwrkwargs['timeframe'] = TFRAMES[args.tframe]
    if args.tann is not None:
        vwrkwargs['tann'] = args.tann
    if args.sigma_max is not None:
        vwrkwargs['sigma_max'] = args.sigma_max
    if args.tau is not None:
        vwrkwargs['tau'] = args.tau
    cerebro.addanalyzer(bt.analyzers.SQN)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A)
    cerebro.addanalyzer(bt.analyzers.VWR, **vwrkwargs)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Months)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Years)
    cerebro.addwriter(bt.WriterFile, csv=args.writercsv, rounding=4)
    cerebro.run()
    if args.plot:
        pkwargs = dict(style='bar')
        if args.plot is not True:
            npkwargs = eval('dict(' + args.plot + ')')
            pkwargs.update(npkwargs)
        cerebro.plot(**pkwargs)

def parse_args(pargs=None):
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='VWR')
    parser.add_argument('--data', '-d', default='../../datas/2005-2006-day-001.txt', help='data to add to the system')
    parser.add_argument('--cash', default=None, type=float, required=False, help='Starting Cash')
    parser.add_argument('--fromdate', '-f', default=None, help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', '-t', default=None, help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--writercsv', '-wcsv', action='store_true', help='Tell the writer to produce a csv stream')
    parser.add_argument('--tframe', '--timeframe', default=None, required=False, choices=TFRAMES.keys(), help='TimeFrame for the Returns/Sharpe calculations')
    parser.add_argument('--sigma-max', required=False, action='store', type=float, default=None, help='VWR Sigma Max')
    parser.add_argument('--tau', required=False, action='store', type=float, default=None, help='VWR tau factor')
    parser.add_argument('--tann', required=False, action='store', type=float, default=None, help='Annualization factor')
    parser.add_argument('--stddev-sample', required=False, action='store_true', help='Consider Bessels correction for stddeviation')
    parser.add_argument('--plot', '-p', nargs='?', required=False, metavar='kwargs', const=True, help='Plot the read data applying any kwargs passed\n\nFor example:\n\n  --plot style="candle" (to plot candles)\n')
    if pargs is not None:
        return parser.parse_args(pargs)
    return parser.parse_args()
if __name__ == '__main__':
    runstrat()