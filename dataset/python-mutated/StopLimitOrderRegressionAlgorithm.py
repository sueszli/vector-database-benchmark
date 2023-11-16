from AlgorithmImports import *

class StopLimitOrderRegressionAlgorithm(QCAlgorithm):
    """Basic algorithm demonstrating how to place stop limit orders."""
    Tolerance = 0.001
    FastPeriod = 30
    SlowPeriod = 60

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 1, 1)
        self.SetEndDate(2017, 1, 1)
        self.SetCash(100000)
        self._symbol = self.AddEquity('SPY', Resolution.Daily).Symbol
        self._fast = self.EMA(self._symbol, self.FastPeriod, Resolution.Daily)
        self._slow = self.EMA(self._symbol, self.SlowPeriod, Resolution.Daily)
        self._buyOrderTicket: OrderTicket = None
        self._sellOrderTicket: OrderTicket = None
        self._previousSlice: Slice = None

    def OnData(self, slice: Slice):
        if False:
            return 10
        if not self.IsReady():
            return
        security = self.Securities[self._symbol]
        if self._buyOrderTicket is None and self.TrendIsUp():
            self._buyOrderTicket = self.StopLimitOrder(self._symbol, 100, stopPrice=security.High * 1.1, limitPrice=security.High * 1.11)
        elif self._buyOrderTicket.Status == OrderStatus.Filled and self._sellOrderTicket is None and self.TrendIsDown():
            self._sellOrderTicket = self.StopLimitOrder(self._symbol, -100, stopPrice=security.Low * 0.99, limitPrice=security.Low * 0.98)

    def OnOrderEvent(self, orderEvent: OrderEvent):
        if False:
            print('Hello World!')
        if orderEvent.Status == OrderStatus.Filled:
            order: StopLimitOrder = self.Transactions.GetOrderById(orderEvent.OrderId)
            if not order.StopTriggered:
                raise Exception('StopLimitOrder StopTriggered should haven been set if the order filled.')
            if orderEvent.Direction == OrderDirection.Buy:
                limitPrice = self._buyOrderTicket.Get(OrderField.LimitPrice)
                if orderEvent.FillPrice > limitPrice:
                    raise Exception(f'Buy stop limit order should have filled with price less than or equal to the limit price {limitPrice}. Fill price: {orderEvent.FillPrice}')
            else:
                limitPrice = self._sellOrderTicket.Get(OrderField.LimitPrice)
                if orderEvent.FillPrice < limitPrice:
                    raise Exception(f'Sell stop limit order should have filled with price greater than or equal to the limit price {limitPrice}. Fill price: {orderEvent.FillPrice}')

    def IsReady(self):
        if False:
            i = 10
            return i + 15
        return self._fast.IsReady and self._slow.IsReady

    def TrendIsUp(self):
        if False:
            return 10
        return self.IsReady() and self._fast.Current.Value > self._slow.Current.Value * (1 + self.Tolerance)

    def TrendIsDown(self):
        if False:
            for i in range(10):
                print('nop')
        return self.IsReady() and self._fast.Current.Value < self._slow.Current.Value * (1 + self.Tolerance)