from AlgorithmImports import *

class BasicTemplateOptionTradesAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)
        self.SetCash(100000)
        option = self.AddOption('GOOG')
        option.SetFilter(-2, +2, 0, 10)
        self.SetBenchmark('GOOG')

    def OnData(self, slice):
        if False:
            for i in range(10):
                print('nop')
        if not self.Portfolio.Invested:
            for kvp in slice.OptionChains:
                chain = kvp.Value
                contracts = sorted(sorted(chain, key=lambda x: abs(chain.Underlying.Price - x.Strike)), key=lambda x: x.Expiry, reverse=False)
                if len(contracts) == 0:
                    continue
                if contracts[0] != None:
                    self.MarketOrder(contracts[0].Symbol, 1)
        else:
            self.Liquidate()
        for kpv in slice.Bars:
            self.Log('---> OnData: {0}, {1}, {2}'.format(self.Time, kpv.Key.Value, str(kpv.Value.Close)))

    def OnOrderEvent(self, orderEvent):
        if False:
            i = 10
            return i + 15
        self.Log(str(orderEvent))