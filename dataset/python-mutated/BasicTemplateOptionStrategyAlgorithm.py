from AlgorithmImports import *

class BasicTemplateOptionStrategyAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetCash(1000000)
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)
        option = self.AddOption('GOOG')
        self.option_symbol = option.Symbol
        option.SetFilter(-2, +2, 0, 180)
        self.SetBenchmark('GOOG')

    def OnData(self, slice):
        if False:
            i = 10
            return i + 15
        if not self.Portfolio.Invested:
            for kvp in slice.OptionChains:
                chain = kvp.Value
                contracts = sorted(sorted(chain, key=lambda x: abs(chain.Underlying.Price - x.Strike)), key=lambda x: x.Expiry, reverse=False)
                if len(contracts) == 0:
                    continue
                atmStraddle = contracts[0]
                if atmStraddle != None:
                    self.Sell(OptionStrategies.Straddle(self.option_symbol, atmStraddle.Strike, atmStraddle.Expiry), 2)
        else:
            self.Liquidate()

    def OnOrderEvent(self, orderEvent):
        if False:
            print('Hello World!')
        self.Log(str(orderEvent))