from AlgorithmImports import *

class ETFConstituentUniverseCompositeDelistingRegressionAlgorithmNoAddEquityETF(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2020, 12, 1)
        self.SetEndDate(2021, 1, 31)
        self.SetCash(100000)
        self.universeSymbolCount = 0
        self.universeAdded = False
        self.universeRemoved = False
        self.UniverseSettings.Resolution = Resolution.Hour
        self.delistingDate = date(2021, 1, 21)
        self.aapl = self.AddEquity('AAPL', Resolution.Hour).Symbol
        self.gdvd = Symbol.Create('GDVD', SecurityType.Equity, Market.USA)
        self.AddUniverse(self.Universe.ETF(self.gdvd, self.UniverseSettings, self.FilterETFs))

    def FilterETFs(self, constituents):
        if False:
            i = 10
            return i + 15
        if self.UtcTime.date() > self.delistingDate:
            raise Exception(f"Performing constituent universe selection on {self.UtcTime.strftime('%Y-%m-%d %H:%M:%S.%f')} after composite ETF has been delisted")
        constituentSymbols = [i.Symbol for i in constituents]
        self.universeSymbolCount = len(constituentSymbols)
        return constituentSymbols

    def OnData(self, data):
        if False:
            print('Hello World!')
        if self.UtcTime.date() > self.delistingDate and any([i != self.aapl for i in data.Keys]):
            raise Exception('Received unexpected slice in OnData(...) after universe was deselected')
        if not self.Portfolio.Invested:
            self.SetHoldings(self.aapl, 0.5)

    def OnSecuritiesChanged(self, changes):
        if False:
            return 10
        if len(changes.AddedSecurities) != 0 and self.UtcTime.date() > self.delistingDate:
            raise Exception('New securities added after ETF constituents were delisted')
        self.universeAdded = self.universeAdded or len(changes.AddedSecurities) >= self.universeSymbolCount
        self.universeRemoved = self.universeRemoved or (len(changes.RemovedSecurities) == self.universeSymbolCount - 1 and self.UtcTime.date() >= self.delistingDate and (self.UtcTime.date() < self.EndDate.date()))

    def OnEndOfAlgorithm(self):
        if False:
            while True:
                i = 10
        if not self.universeAdded:
            raise Exception('ETF constituent universe was never added to the algorithm')
        if not self.universeRemoved:
            raise Exception('ETF constituent universe was not removed from the algorithm after delisting')
        if len(self.ActiveSecurities) > 2:
            raise Exception(f'Expected less than 2 securities after algorithm ended, found {len(self.Securities)}')