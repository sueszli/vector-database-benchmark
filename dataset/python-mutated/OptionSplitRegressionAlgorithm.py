from AlgorithmImports import *

class OptionSplitRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetCash(1000000)
        self.SetStartDate(2014, 6, 6)
        self.SetEndDate(2014, 6, 9)
        option = self.AddOption('AAPL')
        option.SetFilter(self.UniverseFunc)
        self.SetBenchmark('AAPL')
        self.contract = None

    def OnData(self, slice):
        if False:
            for i in range(10):
                print('nop')
        if not self.Portfolio.Invested:
            if self.Time.hour > 9 and self.Time.minute > 0:
                for kvp in slice.OptionChains:
                    chain = kvp.Value
                    contracts = filter(lambda x: x.Strike == 650 and x.Right == OptionRight.Call, chain)
                    sorted_contracts = sorted(contracts, key=lambda x: x.Expiry)
                if len(sorted_contracts) > 1:
                    self.contract = sorted_contracts[1]
                    self.Buy(self.contract.Symbol, 1)
        elif self.Time.day > 6 and self.Time.hour > 14 and (self.Time.minute > 0):
            self.Liquidate()
        if self.Portfolio.Invested:
            options_hold = [x for x in self.Portfolio.Securities if x.Value.Holdings.AbsoluteQuantity != 0]
            holdings = options_hold[0].Value.Holdings.AbsoluteQuantity
            if self.Time.day == 6 and holdings != 1:
                self.Log('Expected position quantity of 1 but was {0}'.format(holdings))
            if self.Time.day == 9 and holdings != 7:
                self.Log('Expected position quantity of 7 but was {0}'.format(holdings))

    def UniverseFunc(self, universe):
        if False:
            while True:
                i = 10
        return universe.IncludeWeeklys().Strikes(-2, 2).Expiration(timedelta(0), timedelta(365 * 2))

    def OnOrderEvent(self, orderEvent):
        if False:
            while True:
                i = 10
        self.Log(str(orderEvent))