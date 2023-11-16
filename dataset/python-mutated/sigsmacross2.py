from datetime import datetime
import backtrader as bt

class SmaCross(bt.SignalStrategy):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        sma1 = bt.ind.SMA(period=10)
        sma2 = bt.ind.SMA(period=30)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)
cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)
data0 = bt.feeds.YahooFinanceData(dataname='YHOO', fromdate=datetime(2011, 1, 1), todate=datetime(2012, 12, 31))
cerebro.adddata(data0)
cerebro.run()
cerebro.plot()