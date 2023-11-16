from AlgorithmImports import *

class CoarseFineFundamentalRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2014, 3, 24)
        self.SetEndDate(2014, 4, 7)
        self.SetCash(50000)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.changes = None
        self.numberOfSymbolsFine = 2

    def CoarseSelectionFunction(self, coarse):
        if False:
            while True:
                i = 10
        tickers = ['GOOG', 'BAC', 'SPY']
        if self.Time.date() < date(2014, 4, 1):
            tickers = ['AAPL', 'AIG', 'IBM']
        return [Symbol.Create(x, SecurityType.Equity, Market.USA) for x in tickers]

    def FineSelectionFunction(self, fine):
        if False:
            while True:
                i = 10
        sortedByMarketCap = sorted(fine, key=lambda x: x.MarketCap, reverse=True)
        return [x.Symbol for x in sortedByMarketCap[:self.numberOfSymbolsFine]]

    def OnData(self, data):
        if False:
            while True:
                i = 10
        if self.changes is None:
            return
        for security in self.changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
                self.Debug('Liquidated Stock: ' + str(security.Symbol.Value))
        for security in self.changes.AddedSecurities:
            if security.Fundamentals.EarningRatios.EquityPerShareGrowth.OneYear > 0.25:
                self.SetHoldings(security.Symbol, 0.5)
                self.Debug('Purchased Stock: ' + str(security.Symbol.Value))
        self.changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            return 10
        self.changes = changes