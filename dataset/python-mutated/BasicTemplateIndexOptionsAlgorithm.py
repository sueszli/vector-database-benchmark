from AlgorithmImports import *

class BasicTemplateIndexOptionsAlgorithm(QCAlgorithm):

    def Initialize(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2021, 1, 4)
        self.SetEndDate(2021, 2, 1)
        self.SetCash(1000000)
        self.spx = self.AddIndex('SPX', Resolution.Minute).Symbol
        spxOptions = self.AddIndexOption(self.spx, Resolution.Minute)
        spxOptions.SetFilter(lambda x: x.CallsOnly())
        self.emaSlow = self.EMA(self.spx, 80)
        self.emaFast = self.EMA(self.spx, 200)

    def OnData(self, data: Slice) -> None:
        if False:
            return 10
        if self.spx not in data.Bars or not self.emaSlow.IsReady:
            return
        for chain in data.OptionChains.Values:
            for contract in chain.Contracts.Values:
                if self.Portfolio.Invested:
                    continue
                if self.emaFast > self.emaSlow and contract.Right == OptionRight.Call or (self.emaFast < self.emaSlow and contract.Right == OptionRight.Put):
                    self.Liquidate(self.InvertOption(contract.Symbol))
                    self.MarketOrder(contract.Symbol, 1)

    def OnEndOfAlgorithm(self) -> None:
        if False:
            while True:
                i = 10
        if self.Portfolio[self.spx].TotalSaleVolume > 0:
            raise Exception('Index is not tradable.')
        if self.Portfolio.TotalSaleVolume == 0:
            raise Exception('Trade volume should be greater than zero by the end of this algorithm')

    def InvertOption(self, symbol: Symbol) -> Symbol:
        if False:
            print('Hello World!')
        return Symbol.CreateOption(symbol.Underlying, symbol.ID.Market, symbol.ID.OptionStyle, OptionRight.Put if symbol.ID.OptionRight == OptionRight.Call else OptionRight.Call, symbol.ID.StrikePrice, symbol.ID.Date)