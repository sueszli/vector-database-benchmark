from AlgorithmImports import *
from QuantConnect.Data.Custom.IconicTypes import *

class CustomDataLinkedIconicTypeAddDataOnSecuritiesChangedRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2014, 3, 24)
        self.SetEndDate(2014, 4, 7)
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverseSelection(CoarseFundamentalUniverseSelectionModel(self.CoarseSelector))

    def CoarseSelector(self, coarse):
        if False:
            i = 10
            return i + 15
        return [Symbol.Create('AAPL', SecurityType.Equity, Market.USA), Symbol.Create('BAC', SecurityType.Equity, Market.USA), Symbol.Create('FB', SecurityType.Equity, Market.USA), Symbol.Create('GOOGL', SecurityType.Equity, Market.USA), Symbol.Create('GOOG', SecurityType.Equity, Market.USA), Symbol.Create('IBM', SecurityType.Equity, Market.USA)]

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        if not self.Portfolio.Invested and len(self.Transactions.GetOpenOrders()) == 0:
            aapl = Symbol.Create('AAPL', SecurityType.Equity, Market.USA)
            self.SetHoldings(aapl, 0.5)
        for customSymbol in self.customSymbols:
            if not self.ActiveSecurities.ContainsKey(customSymbol.Underlying):
                raise Exception(f'Custom data undelrying ({customSymbol.Underlying}) Symbol was not found in active securities')

    def OnSecuritiesChanged(self, changes):
        if False:
            print('Hello World!')
        iterated = False
        for added in changes.AddedSecurities:
            if not iterated:
                self.customSymbols = []
                iterated = True
            self.customSymbols.append(self.AddData(LinkedData, added.Symbol, Resolution.Daily).Symbol)