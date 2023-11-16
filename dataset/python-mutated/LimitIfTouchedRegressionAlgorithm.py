from AlgorithmImports import *
from collections import deque

class LimitIfTouchedRegressionAlgorithm(QCAlgorithm):
    _expectedEvents = deque(['Time: 10/10/2013 13:31:00 OrderID: 72 EventID: 399 Symbol: SPY Status: Filled Quantity: -1 FillQuantity: -1 FillPrice: 144.6434 USD LimitPrice: 144.3551 TriggerPrice: 143.61 OrderFee: 1 USD', 'Time: 10/10/2013 15:57:00 OrderID: 73 EventID: 156 Symbol: SPY Status: Filled Quantity: -1 FillQuantity: -1 FillPrice: 145.6636 USD LimitPrice: 145.6434 TriggerPrice: 144.89 OrderFee: 1 USD', 'Time: 10/11/2013 15:37:00 OrderID: 74 EventID: 380 Symbol: SPY Status: Filled Quantity: -1 FillQuantity: -1 FillPrice: 146.7185 USD LimitPrice: 146.6723 TriggerPrice: 145.92 OrderFee: 1 USD'])

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity('SPY')

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        if data.ContainsKey('SPY'):
            if len(self.Transactions.GetOpenOrders()) == 0:
                self._negative = 1 if self.Time.day < 9 else -1
                orderRequest = SubmitOrderRequest(OrderType.LimitIfTouched, SecurityType.Equity, 'SPY', self._negative * 10, 0, data['SPY'].Price - self._negative, data['SPY'].Price - 0.25 * self._negative, self.UtcTime, f'LIT - Quantity: {self._negative * 10}')
                self._request = self.Transactions.AddOrder(orderRequest)
                return
            if self._request is not None:
                if self._request.Quantity == 1:
                    self.Transactions.CancelOpenOrders()
                    self._request = None
                    return
                new_quantity = int(self._request.Quantity - self._negative)
                self._request.UpdateQuantity(new_quantity, f'LIT - Quantity: {new_quantity}')
                self._request.UpdateTriggerPrice(Extensions.RoundToSignificantDigits(self._request.Get(OrderField.TriggerPrice), 5))

    def OnOrderEvent(self, orderEvent):
        if False:
            for i in range(10):
                print('nop')
        if orderEvent.Status == OrderStatus.Filled:
            expected = self._expectedEvents.popleft()
            if orderEvent.ToString() != expected:
                raise Exception(f'orderEvent {orderEvent.Id} differed from {expected}')