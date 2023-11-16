from AlgorithmImports import *

class FilteredIdentityAlgorithm(QCAlgorithm):
    """ Example algorithm of the Identity indicator with the filtering enhancement """

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2014, 5, 2)
        self.SetEndDate(self.StartDate)
        self.SetCash(100000)
        security = self.AddForex('EURUSD', Resolution.Tick)
        self.symbol = security.Symbol
        self.identity = self.FilteredIdentity(self.symbol, None, self.Filter)

    def Filter(self, data):
        if False:
            print('Hello World!')
        'Filter function: True if data is not an instance of Tick. If it is, true if TickType is Trade\n        data -- Data for applying the filter'
        if isinstance(data, Tick):
            return data.TickType == TickType.Trade
        return True

    def OnData(self, data):
        if False:
            print('Hello World!')
        if not self.identity.IsReady:
            return
        if not self.Portfolio.Invested:
            self.SetHoldings(self.symbol, 1)
            self.Debug('Purchased Stock')