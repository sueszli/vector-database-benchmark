from AlgorithmImports import *

class StatefulCoarseUniverseSelectionBenchmark(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2017, 1, 1)
        self.SetEndDate(2019, 1, 1)
        self.SetCash(50000)
        self.AddUniverse(self.CoarseSelectionFunction)
        self.numberOfSymbols = 250
        self._blackList = []

    def CoarseSelectionFunction(self, coarse):
        if False:
            while True:
                i = 10
        selected = [x for x in coarse if x.HasFundamentalData]
        sortedByDollarVolume = sorted(selected, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sortedByDollarVolume[:self.numberOfSymbols] if not x.Symbol in self._blackList]

    def OnData(self, slice):
        if False:
            print('Hello World!')
        if slice.HasData:
            symbol = slice.Keys[0]
            if symbol:
                if len(self._blackList) > 50:
                    self._blackList.pop(0)
                self._blackList.append(symbol)

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