from AlgorithmImports import *

class CustomMarginInterestRateModelAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 1)
        self.SetEndDate(2013, 10, 31)
        security = self.AddEquity('SPY', Resolution.Hour)
        self._spy = security.Symbol
        self._marginInterestRateModel = CustomMarginInterestRateModel()
        security.SetMarginInterestRateModel(self._marginInterestRateModel)
        self._cashAfterOrder = 0

    def OnData(self, data: Slice):
        if False:
            return 10
        if not self.Portfolio.Invested:
            self.SetHoldings(self._spy, 1)

    def OnOrderEvent(self, orderEvent: OrderEvent):
        if False:
            for i in range(10):
                print('nop')
        if orderEvent.Status == OrderStatus.Filled:
            self._cashAfterOrder = self.Portfolio.Cash

    def OnEndOfAlgorithm(self):
        if False:
            while True:
                i = 10
        if self._marginInterestRateModel.callCount == 0:
            raise Exception('CustomMarginInterestRateModel was not called')
        expectedCash = self._cashAfterOrder * pow(1 + self._marginInterestRateModel.interestRate, self._marginInterestRateModel.callCount)
        if abs(self.Portfolio.Cash - expectedCash) > 1e-10:
            raise Exception(f'Expected cash {expectedCash} but got {self.Portfolio.Cash}')

class CustomMarginInterestRateModel:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.interestRate = 0.01
        self.callCount = 0

    def ApplyMarginInterestRate(self, parameters: MarginInterestRateParameters):
        if False:
            print('Hello World!')
        security = parameters.Security
        positionValue = security.Holdings.GetQuantityValue(security.Holdings.Quantity)
        if positionValue.Amount > 0:
            positionValue.Cash.AddAmount(self.interestRate * positionValue.Cash.Amount)
            self.callCount += 1