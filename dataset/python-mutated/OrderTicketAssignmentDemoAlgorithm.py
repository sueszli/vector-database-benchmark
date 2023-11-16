from datetime import timedelta
from AlgorithmImports import *

class OrderTicketAssignmentDemoAlgorithm(QCAlgorithm):
    """Demonstration on how to access order tickets right after placing an order."""

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.symbol = self.AddEquity('SPY').Symbol
        self.trade_count = 0
        self.Consolidate(self.symbol, timedelta(hours=1), self.HourConsolidator)

    def HourConsolidator(self, bar: TradeBar):
        if False:
            for i in range(10):
                print('nop')
        self.ticket = None
        self.ticket = self.MarketOrder(self.symbol, 1, asynchronous=True)
        self.Debug(f'{self.Time}: Buy: Price {bar.Price}, orderId: {self.ticket.OrderId}')
        self.trade_count += 1

    def OnOrderEvent(self, orderEvent: OrderEvent):
        if False:
            for i in range(10):
                print('nop')
        ticket = orderEvent.Ticket
        if ticket is None:
            raise Exception('Expected order ticket in order event to not be null')
        if orderEvent.Status == OrderStatus.Submitted and self.ticket is not None:
            raise Exception('Field self.ticket not expected no be assigned on the first order event')
        self.Debug(ticket.ToString())

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.Portfolio.Invested or self.trade_count != self.Transactions.OrdersCount:
            raise Exception(f'Expected the portfolio to have holdings and to have {self.tradeCount} trades, but had {self.Transactions.OrdersCount}')