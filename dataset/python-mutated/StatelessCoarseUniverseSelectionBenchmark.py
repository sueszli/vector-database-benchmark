from AlgorithmImports import *

class StatelessCoarseUniverseSelectionBenchmark(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2017, 1, 1)
        self.SetEndDate(2019, 1, 1)
        self.SetCash(50000)
        self.AddUniverse(self.CoarseSelectionFunction)
        self.numberOfSymbols = 250

    def CoarseSelectionFunction(self, coarse):
        if False:
            i = 10
            return i + 15
        selected = [x for x in coarse if x.HasFundamentalData]
        sortedByDollarVolume = sorted(selected, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sortedByDollarVolume[:self.numberOfSymbols]]

    def OnSecuritiesChanged(self, changes):
        if False:
            print('Hello World!')
        if changes is None:
            return
        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
        for security in changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 0.001)