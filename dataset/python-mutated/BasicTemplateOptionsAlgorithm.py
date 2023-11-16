from AlgorithmImports import *

class BasicTemplateOptionsAlgorithm(QCAlgorithm):
    UnderlyingTicker = 'GOOG'

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)
        self.SetCash(100000)
        equity = self.AddEquity(self.UnderlyingTicker)
        option = self.AddOption(self.UnderlyingTicker)
        self.option_symbol = option.Symbol
        option.SetFilter(lambda u: u.Strikes(-2, +2).Expiration(0, 180))
        self.SetBenchmark(equity.Symbol)

    def OnData(self, slice):
        if False:
            print('Hello World!')
        if self.Portfolio.Invested or not self.IsMarketOpen(self.option_symbol):
            return
        chain = slice.OptionChains.GetValue(self.option_symbol)
        if chain is None:
            return
        contracts = sorted(sorted(sorted(chain, key=lambda x: abs(chain.Underlying.Price - x.Strike)), key=lambda x: x.Expiry, reverse=True), key=lambda x: x.Right, reverse=True)
        if len(contracts) == 0:
            return
        symbol = contracts[0].Symbol
        self.MarketOrder(symbol, 1)
        self.MarketOnCloseOrder(symbol, -1)

    def OnOrderEvent(self, orderEvent):
        if False:
            while True:
                i = 10
        self.Log(str(orderEvent))