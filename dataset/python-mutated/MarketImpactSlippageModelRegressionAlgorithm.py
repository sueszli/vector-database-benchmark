from AlgorithmImports import *

class MarketImpactSlippageModelRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 13)
        self.SetCash(10000000)
        spy = self.AddEquity('SPY', Resolution.Daily)
        aapl = self.AddEquity('AAPL', Resolution.Daily)
        spy.SetSlippageModel(MarketImpactSlippageModel(self))
        aapl.SetSlippageModel(MarketImpactSlippageModel(self))

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        self.SetHoldings('SPY', 0.5)
        self.SetHoldings('AAPL', -0.5)

    def OnOrderEvent(self, orderEvent):
        if False:
            print('Hello World!')
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f'Price: {self.Securities[orderEvent.Symbol].Price}, filled price: {orderEvent.FillPrice}, quantity: {orderEvent.FillQuantity}')