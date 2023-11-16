from AlgorithmImports import *

class CoarseFineUniverseSelectionBenchmark(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2017, 11, 1)
        self.SetEndDate(2018, 3, 1)
        self.SetCash(50000)
        self.UniverseSettings.Resolution = Resolution.Minute
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.numberOfSymbols = 150
        self.numberOfSymbolsFine = 40
        self._changes = None

    def CoarseSelectionFunction(self, coarse):
        if False:
            return 10
        selected = [x for x in coarse if x.HasFundamentalData]
        sortedByDollarVolume = sorted(selected, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sortedByDollarVolume[:self.numberOfSymbols]]

    def FineSelectionFunction(self, fine):
        if False:
            while True:
                i = 10
        sortedByPeRatio = sorted(fine, key=lambda x: x.ValuationRatios.PERatio, reverse=True)
        return [x.Symbol for x in sortedByPeRatio[:self.numberOfSymbolsFine]]

    def OnData(self, data):
        if False:
            return 10
        if self._changes is None:
            return
        for security in self._changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
        for security in self._changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 0.02)
        self._changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            i = 10
            return i + 15
        self._changes = changes