from AlgorithmImports import *

class EmaCrossFuturesFrontMonthAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 10)
        self.SetCash(1000000)
        future = self.AddFuture(Futures.Metals.Gold)
        future.SetFilter(lambda x: x.FrontMonth().OnlyApplyFilterAtMarketOpen())
        self.symbol = None
        self.fast = ExponentialMovingAverage(100)
        self.slow = ExponentialMovingAverage(300)
        self.tolerance = 0.001
        self.consolidator = None
        chart = Chart('EMA Cross')
        chart.AddSeries(Series('Fast', SeriesType.Line, 0))
        chart.AddSeries(Series('Slow', SeriesType.Line, 0))
        self.AddChart(chart)

    def OnData(self, slice):
        if False:
            return 10
        holding = None if self.symbol is None else self.Portfolio.get(self.symbol)
        if holding is not None:
            if self.fast.Current.Value > self.slow.Current.Value * (1 + self.tolerance):
                if not holding.Invested:
                    self.SetHoldings(self.symbol, 0.1)
                    self.PlotEma()
            elif holding.Invested:
                self.Liquidate(self.symbol)
                self.PlotEma()

    def OnSecuritiesChanged(self, changes):
        if False:
            while True:
                i = 10
        if len(changes.RemovedSecurities) > 0:
            if self.symbol is not None and self.consolidator is not None:
                self.SubscriptionManager.RemoveConsolidator(self.symbol, self.consolidator)
                self.fast.Reset()
                self.slow.Reset()
        self.symbol = changes.AddedSecurities[0].Symbol
        self.consolidator = self.ResolveConsolidator(self.symbol, Resolution.Minute)
        self.RegisterIndicator(self.symbol, self.fast, self.consolidator)
        self.RegisterIndicator(self.symbol, self.slow, self.consolidator)
        self.WarmUpIndicator(self.symbol, self.fast, Resolution.Minute)
        self.WarmUpIndicator(self.symbol, self.slow, Resolution.Minute)
        self.PlotEma()

    def PlotEma(self):
        if False:
            for i in range(10):
                print('nop')
        self.Plot('EMA Cross', 'Fast', self.fast.Current.Value)
        self.Plot('EMA Cross', 'Slow', self.slow.Current.Value)