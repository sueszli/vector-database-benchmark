from AlgorithmImports import *

class TimeInForceAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.symbol = self.AddEquity('SPY', Resolution.Minute).Symbol
        self.gtcOrderTicket1 = None
        self.gtcOrderTicket2 = None
        self.dayOrderTicket1 = None
        self.dayOrderTicket2 = None
        self.gtdOrderTicket1 = None
        self.gtdOrderTicket2 = None
        self.expectedOrderStatuses = {}

    def OnData(self, data):
        if False:
            while True:
                i = 10
        if self.gtcOrderTicket1 is None:
            self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
            self.gtcOrderTicket1 = self.LimitOrder(self.symbol, 10, 100)
            self.expectedOrderStatuses[self.gtcOrderTicket1.OrderId] = OrderStatus.Submitted
            self.gtcOrderTicket2 = self.LimitOrder(self.symbol, 10, 160)
            self.expectedOrderStatuses[self.gtcOrderTicket2.OrderId] = OrderStatus.Filled
        if self.dayOrderTicket1 is None:
            self.DefaultOrderProperties.TimeInForce = TimeInForce.Day
            self.dayOrderTicket1 = self.LimitOrder(self.symbol, 10, 140)
            self.expectedOrderStatuses[self.dayOrderTicket1.OrderId] = OrderStatus.Canceled
            self.dayOrderTicket2 = self.LimitOrder(self.symbol, 10, 180)
            self.expectedOrderStatuses[self.dayOrderTicket2.OrderId] = OrderStatus.Filled
        if self.gtdOrderTicket1 is None:
            self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilDate(datetime(2013, 10, 10))
            self.gtdOrderTicket1 = self.LimitOrder(self.symbol, 10, 100)
            self.expectedOrderStatuses[self.gtdOrderTicket1.OrderId] = OrderStatus.Canceled
            self.gtdOrderTicket2 = self.LimitOrder(self.symbol, 10, 160)
            self.expectedOrderStatuses[self.gtdOrderTicket2.OrderId] = OrderStatus.Filled

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        self.Debug(f'{self.Time} {orderEvent}')

    def OnEndOfAlgorithm(self):
        if False:
            i = 10
            return i + 15
        for (orderId, expectedStatus) in self.expectedOrderStatuses.items():
            order = self.Transactions.GetOrderById(orderId)
            if order.Status != expectedStatus:
                raise Exception(f'Invalid status for order {orderId} - Expected: {expectedStatus}, actual: {order.Status}')