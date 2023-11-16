from AlgorithmImports import *

class OptionExerciseAssignRegressionAlgorithm(QCAlgorithm):
    UnderlyingTicker = 'GOOG'

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetCash(100000)
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 28)
        self.equity = self.AddEquity(self.UnderlyingTicker)
        self.option = self.AddOption(self.UnderlyingTicker)
        self.option.SetFilter(self.UniverseFunc)
        self.SetBenchmark(self.equity.Symbol)
        self._assignedOption = False

    def OnData(self, slice):
        if False:
            i = 10
            return i + 15
        if self.Portfolio.Invested:
            return
        for kvp in slice.OptionChains:
            chain = kvp.Value
            contracts = filter(lambda x: x.Expiry.date() == self.Time.date() and x.Strike < chain.Underlying.Price and (x.Right == OptionRight.Call), chain)
            sorted_contracts = sorted(contracts, key=lambda x: x.Strike, reverse=True)[:2]
            if sorted_contracts:
                self.MarketOrder(sorted_contracts[0].Symbol, 1)
                self.MarketOrder(sorted_contracts[1].Symbol, -1)

    def UniverseFunc(self, universe):
        if False:
            for i in range(10):
                print('nop')
        return universe.IncludeWeeklys().Strikes(-2, 2).Expiration(timedelta(0), timedelta(10))

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        self.Log(str(orderEvent))

    def OnAssignmentOrderEvent(self, assignmentEvent):
        if False:
            print('Hello World!')
        self.Log(str(assignmentEvent))
        self._assignedOption = True