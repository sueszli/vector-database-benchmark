from cmath import isclose
from AlgorithmImports import *

class SecurityCustomPropertiesAlgorithm(QCAlgorithm):
    """Demonstration of how to use custom security properties.
    In this algorithm we trade a security based on the values of a slow and fast EMAs which are stored in the security itself."""

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.spy = self.AddEquity('SPY', Resolution.Minute)
        self.spy.SlowEma = self.EMA(self.spy.Symbol, 30, Resolution.Minute)
        self.spy.Add('FastEma', self.EMA(self.spy.Symbol, 60, Resolution.Minute))
        self.spy['BB'] = self.BB(self.spy.Symbol, 20, 1, MovingAverageType.Simple, Resolution.Minute)
        self.spy.FeeFactor = 2e-05
        self.spy.SetFeeModel(CustomFeeModel())
        self.spy.OrdersFeesPrices = {}

    def OnData(self, data):
        if False:
            return 10
        if not self.spy.FastEma.IsReady:
            return
        if not self.Portfolio.Invested:
            if self.spy.SlowEma > self.spy.FastEma:
                self.SetHoldings(self.spy.Symbol, 1)
        elif self.spy.Get[ExponentialMovingAverage]('SlowEma') < self.spy.Get[ExponentialMovingAverage]('FastEma'):
            self.Liquidate(self.spy.Symbol)
        bb: BollingerBands = self.spy['BB']
        self.Plot('BB', bb.UpperBand, bb.MiddleBand, bb.LowerBand)

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        if orderEvent.Status == OrderStatus.Filled:
            fee = orderEvent.OrderFee
            expectedFee = self.spy.OrdersFeesPrices[orderEvent.OrderId] * orderEvent.AbsoluteFillQuantity * self.spy.FeeFactor
            if not isclose(fee.Value.Amount, expectedFee, rel_tol=1e-15):
                raise Exception(f'Custom fee model failed to set the correct fee. Expected: {expectedFee}. Actual: {fee.Value.Amount}')

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        if self.Transactions.OrdersCount == 0:
            raise Exception('No orders executed')

class CustomFeeModel(FeeModel):
    """This custom fee is implemented for demonstration purposes only."""

    def GetOrderFee(self, parameters):
        if False:
            return 10
        security = parameters.Security
        feeFactor = security.FeeFactor
        if feeFactor is None:
            feeFactor = 1e-05
        security['OrdersFeesPrices'][parameters.Order.Id] = security.Price
        fee = max(1.0, security.Price * parameters.Order.AbsoluteQuantity * feeFactor)
        return OrderFee(CashAmount(fee, 'USD'))