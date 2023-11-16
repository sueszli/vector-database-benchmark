from AlgorithmImports import *

class ConsolidateDifferentTickTypesRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 10, 6)
        self.SetEndDate(2013, 10, 7)
        self.SetCash(1000000)
        equity = self.AddEquity('SPY', Resolution.Tick, Market.USA)
        quoteConsolidator = self.Consolidate(equity.Symbol, Resolution.Tick, TickType.Quote, lambda tick: self.OnQuoteTick(tick))
        self.thereIsAtLeastOneQuoteTick = False
        tradeConsolidator = self.Consolidate(equity.Symbol, Resolution.Tick, TickType.Trade, lambda tick: self.OnTradeTick(tick))
        self.thereIsAtLeastOneTradeTick = False

    def OnQuoteTick(self, tick):
        if False:
            i = 10
            return i + 15
        self.thereIsAtLeastOneQuoteTick = True
        if tick.TickType != TickType.Quote:
            raise Exception(f'The type of the tick should be Quote, but was {tick.TickType}')

    def OnTradeTick(self, tick):
        if False:
            while True:
                i = 10
        self.thereIsAtLeastOneTradeTick = True
        if tick.TickType != TickType.Trade:
            raise Exception(f'The type of the tick should be Trade, but was {tick.TickType}')

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        if not self.thereIsAtLeastOneQuoteTick:
            raise Exception(f"There should have been at least one tick in OnQuoteTick() method, but there wasn't")
        if not self.thereIsAtLeastOneTradeTick:
            raise Exception(f"There should have been at least one tick in OnTradeTick() method, but there wasn't")