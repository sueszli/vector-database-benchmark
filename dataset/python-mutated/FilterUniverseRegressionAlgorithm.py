from AlgorithmImports import *

class FilterUniverseRegressionAlgorithm(QCAlgorithm):
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
            return 10
        universe = universe.WeeklysOnly().Strikes(-5, +5).CallsOnly().Expiration(0, 1)
        return universe

    def OnData(self, slice):
        if False:
            i = 10
            return i + 15
        if self.Portfolio.Invested:
            return
        for kvp in slice.OptionChains:
            if kvp.Key != self.OptionSymbol:
                continue
            chain = kvp.Value
            contracts = [option for option in sorted(chain, key=lambda x: x.Strike, reverse=True)]
            if contracts:
                self.MarketOrder(contracts[0].Symbol, 1)