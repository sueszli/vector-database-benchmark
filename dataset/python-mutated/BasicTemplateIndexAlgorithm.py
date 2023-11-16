from AlgorithmImports import *

class BasicTemplateIndexAlgorithm(QCAlgorithm):

    def Initialize(self) -> None:
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2021, 1, 4)
        self.SetEndDate(2021, 1, 18)
        self.SetCash(1000000)
        self.spx = self.AddIndex('SPX', Resolution.Minute).Symbol
        self.spxOption = Symbol.CreateOption(self.spx, Market.USA, OptionStyle.European, OptionRight.Call, 3200, datetime(2021, 1, 15))
        self.AddIndexOptionContract(self.spxOption, Resolution.Minute)
        self.emaSlow = self.EMA(self.spx, 80)
        self.emaFast = self.EMA(self.spx, 200)

    def OnData(self, data: Slice):
        if False:
            print('Hello World!')
        if self.spx not in data.Bars or self.spxOption not in data.Bars:
            return
        if not self.emaSlow.IsReady:
            return
        if self.emaFast > self.emaSlow:
            self.SetHoldings(self.spxOption, 1)
        else:
            self.Liquidate()

    def OnEndOfAlgorithm(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.Portfolio[self.spx].TotalSaleVolume > 0:
            raise Exception('Index is not tradable.')