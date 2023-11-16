from AlgorithmImports import *

class BasicTemplateOptionsHourlyAlgorithm(QCAlgorithm):
    UnderlyingTicker = 'AAPL'

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2014, 6, 6)
        self.SetEndDate(2014, 6, 9)
        self.SetCash(100000)
        equity = self.AddEquity(self.UnderlyingTicker, Resolution.Hour)
        option = self.AddOption(self.UnderlyingTicker, Resolution.Hour)
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
        if len(contracts) == 0 or not self.IsMarketOpen(contracts[0].Symbol):
            return
        symbol = contracts[0].Symbol
        self.MarketOrder(symbol, 1)
        self.MarketOnCloseOrder(symbol, -1)

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        self.Log(str(orderEvent))