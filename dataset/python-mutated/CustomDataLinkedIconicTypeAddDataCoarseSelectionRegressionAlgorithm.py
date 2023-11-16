from AlgorithmImports import *
from QuantConnect.Data.Custom.IconicTypes import *

class CustomDataLinkedIconicTypeAddDataCoarseSelectionRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2014, 3, 24)
        self.SetEndDate(2014, 4, 7)
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverseSelection(CoarseFundamentalUniverseSelectionModel(self.CoarseSelector))

    def CoarseSelector(self, coarse):
        if False:
            print('Hello World!')
        symbols = [Symbol.Create('AAPL', SecurityType.Equity, Market.USA), Symbol.Create('BAC', SecurityType.Equity, Market.USA), Symbol.Create('FB', SecurityType.Equity, Market.USA), Symbol.Create('GOOGL', SecurityType.Equity, Market.USA), Symbol.Create('GOOG', SecurityType.Equity, Market.USA), Symbol.Create('IBM', SecurityType.Equity, Market.USA)]
        self.customSymbols = []
        for symbol in symbols:
            self.customSymbols.append(self.AddData(LinkedData, symbol, Resolution.Daily).Symbol)
        return symbols

    def OnData(self, data):
        if False:
            print('Hello World!')
        if not self.Portfolio.Invested and len(self.Transactions.GetOpenOrders()) == 0:
            aapl = Symbol.Create('AAPL', SecurityType.Equity, Market.USA)
            self.SetHoldings(aapl, 0.5)
        for customSymbol in self.customSymbols:
            if not self.ActiveSecurities.ContainsKey(customSymbol.Underlying):
                raise Exception(f'Custom data undelrying ({customSymbol.Underlying}) Symbol was not found in active securities')