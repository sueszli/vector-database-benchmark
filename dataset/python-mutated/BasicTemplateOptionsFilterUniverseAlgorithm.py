from AlgorithmImports import *

class BasicTemplateOptionsFilterUniverseAlgorithm(QCAlgorithm):
    UnderlyingTicker = 'GOOG'

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 28)
        self.SetCash(100000)
        equity = self.AddEquity(self.UnderlyingTicker)
        option = self.AddOption(self.UnderlyingTicker)
        self.OptionSymbol = option.Symbol
        option.SetFilter(self.FilterFunction)
        self.SetBenchmark(equity.Symbol)

    def FilterFunction(self, universe):
        if False:
            while True:
                i = 10
        universe = universe.WeeklysOnly().Expiration(0, 1)
        return [symbol for symbol in universe if symbol.ID.OptionRight != OptionRight.Put and -10 < universe.Underlying.Price - symbol.ID.StrikePrice < 10]

    def OnData(self, slice):
        if False:
            return 10
        if self.Portfolio.Invested:
            return
        for kvp in slice.OptionChains:
            if kvp.Key != self.OptionSymbol:
                continue
            chain = kvp.Value
            contracts = [option for option in sorted(chain, key=lambda x: x.Strike, reverse=True) if option.Expiry.date() == self.Time.date() and option.Strike < chain.Underlying.Price]
            if contracts:
                self.MarketOrder(contracts[0].Symbol, 1)