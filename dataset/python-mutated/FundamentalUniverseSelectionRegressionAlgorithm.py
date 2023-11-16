from AlgorithmImports import *
from Selection.FundamentalUniverseSelectionModel import FundamentalUniverseSelectionModel

class FundamentalUniverseSelectionRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2014, 3, 25)
        self.SetEndDate(2014, 4, 7)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddEquity('SPY')
        self.AddEquity('AAPL')
        self.SetUniverseSelection(FundamentalUniverseSelectionModelTest())
        self.changes = None

    def SelectionFunction(self, fundamental):
        if False:
            print('Hello World!')
        sortedByDollarVolume = sorted([x for x in fundamental if x.Price > 1], key=lambda x: x.DollarVolume, reverse=True)
        sortedByPeRatio = sorted(sortedByDollarVolume, key=lambda x: x.ValuationRatios.PERatio, reverse=True)
        return [x.Symbol for x in sortedByPeRatio[:self.numberOfSymbolsFundamental]]

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
            self.SetHoldings(security.Symbol, 0.02)
        self.changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            while True:
                i = 10
        self.changes = changes

class FundamentalUniverseSelectionModelTest(FundamentalUniverseSelectionModel):

    def Select(self, algorithm, fundamental):
        if False:
            for i in range(10):
                print('nop')
        sortedByDollarVolume = sorted([x for x in fundamental if x.HasFundamentalData and x.Price > 1], key=lambda x: x.DollarVolume, reverse=True)
        sortedByPeRatio = sorted(sortedByDollarVolume, key=lambda x: x.ValuationRatios.PERatio, reverse=True)
        return [x.Symbol for x in sortedByPeRatio[:2]]