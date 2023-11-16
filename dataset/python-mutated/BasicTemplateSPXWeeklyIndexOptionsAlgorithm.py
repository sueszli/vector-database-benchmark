from AlgorithmImports import *

class BasicTemplateSPXWeeklyIndexOptionsAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2021, 1, 4)
        self.SetEndDate(2021, 1, 10)
        self.SetCash(1000000)
        self.spx = self.AddIndex('SPX').Symbol
        self.spxOptions = self.AddIndexOption(self.spx)
        self.spxOptions.SetFilter(lambda u: u.Strikes(0, 1).Expiration(0, 30))
        spxw = self.AddIndexOption(self.spx, 'SPXW')
        spxw.SetFilter(lambda u: u.Strikes(0, 1).Expiration(0, 7).IncludeWeeklys())
        self.spxw_option = spxw.Symbol

    def OnData(self, slice):
        if False:
            while True:
                i = 10
        if self.Portfolio.Invested:
            return
        chain = slice.OptionChains.GetValue(self.spxw_option)
        if chain is None:
            return
        contracts = sorted(sorted(sorted(chain, key=lambda x: x.Expiry), key=lambda x: abs(chain.Underlying.Price - x.Strike)), key=lambda x: x.Right, reverse=True)
        if len(contracts) == 0:
            return
        symbol = contracts[0].Symbol
        self.MarketOrder(symbol, 1)

    def OnOrderEvent(self, orderEvent):
        if False:
            while True:
                i = 10
        self.Debug(str(orderEvent))