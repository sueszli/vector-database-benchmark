from AlgorithmImports import *

class OptionChainConsistencyRegressionAlgorithm(QCAlgorithm):
    UnderlyingTicker = 'GOOG'

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetCash(10000)
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)
        self.equity = self.AddEquity(self.UnderlyingTicker)
        self.option = self.AddOption(self.UnderlyingTicker)
        self.option.SetFilter(self.UniverseFunc)
        self.SetBenchmark(self.equity.Symbol)

    def OnData(self, slice):
        if False:
            i = 10
            return i + 15
        if self.Portfolio.Invested:
            return
        for kvp in slice.OptionChains:
            chain = kvp.Value
            for o in chain:
                if not self.Securities.ContainsKey(o.Symbol):
                    self.Log('Inconsistency found: option chains contains contract {0} that is not available in securities manager and not available for trading'.format(o.Symbol.Value))
            contracts = filter(lambda x: x.Expiry.date() == self.Time.date() and x.Strike < chain.Underlying.Price and (x.Right == OptionRight.Call), chain)
            sorted_contracts = sorted(contracts, key=lambda x: x.Strike, reverse=True)
            if len(sorted_contracts) > 2:
                self.MarketOrder(sorted_contracts[2].Symbol, 1)
                self.MarketOnCloseOrder(sorted_contracts[2].Symbol, -1)

    def UniverseFunc(self, universe):
        if False:
            return 10
        return universe.IncludeWeeklys().Strikes(-2, 2).Expiration(timedelta(0), timedelta(10))

    def OnOrderEvent(self, orderEvent):
        if False:
            i = 10
            return i + 15
        self.Log(str(orderEvent))