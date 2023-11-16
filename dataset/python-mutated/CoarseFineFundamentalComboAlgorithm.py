from AlgorithmImports import *
from System.Collections.Generic import List

class CoarseFineFundamentalComboAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2014, 1, 1)
        self.SetEndDate(2015, 1, 1)
        self.SetCash(50000)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.__numberOfSymbols = 5
        self.__numberOfSymbolsFine = 2
        self._changes = None

    def CoarseSelectionFunction(self, coarse):
        if False:
            print('Hello World!')
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sortedByDollarVolume[:self.__numberOfSymbols]]

    def FineSelectionFunction(self, fine):
        if False:
            i = 10
            return i + 15
        sortedByPeRatio = sorted(fine, key=lambda x: x.ValuationRatios.PERatio, reverse=True)
        return [x.Symbol for x in sortedByPeRatio[:self.__numberOfSymbolsFine]]

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self._changes is None:
            return
        for security in self._changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
        for security in self._changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 0.2)
        self._changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            i = 10
            return i + 15
        self._changes = changes