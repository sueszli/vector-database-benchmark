from AlgorithmImports import *
from math import copysign

class UpdateOrderRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 1, 1)
        self.SetEndDate(2015, 1, 1)
        self.SetCash(100000)
        self.security = self.AddEquity('SPY', Resolution.Daily)
        self.last_month = -1
        self.quantity = 100
        self.delta_quantity = 10
        self.stop_percentage = 0.025
        self.stop_percentage_delta = 0.005
        self.limit_percentage = 0.025
        self.limit_percentage_delta = 0.005
        OrderTypeEnum = [OrderType.Market, OrderType.Limit, OrderType.StopMarket, OrderType.StopLimit, OrderType.MarketOnOpen, OrderType.MarketOnClose, OrderType.TrailingStop]
        self.order_types_queue = CircularQueue[OrderType](OrderTypeEnum)
        self.order_types_queue.CircleCompleted += self.onCircleCompleted
        self.tickets = []

    def onCircleCompleted(self, sender, event):
        if False:
            for i in range(10):
                print('nop')
        "Flip our signs when we've gone through all the order types"
        self.quantity *= -1

    def OnData(self, data):
        if False:
            print('Hello World!')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if not data.ContainsKey('SPY'):
            return
        if self.Time.month != self.last_month:
            orderType = self.order_types_queue.Dequeue()
            self.Log('\r\n--------------MONTH: {0}:: {1}\r\n'.format(self.Time.strftime('%B'), orderType))
            self.last_month = self.Time.month
            self.Log('ORDER TYPE:: {0}'.format(orderType))
            isLong = self.quantity > 0
            stopPrice = (1 + self.stop_percentage) * data['SPY'].High if isLong else (1 - self.stop_percentage) * data['SPY'].Low
            limitPrice = (1 - self.limit_percentage) * stopPrice if isLong else (1 + self.limit_percentage) * stopPrice
            if orderType == OrderType.Limit:
                limitPrice = (1 + self.limit_percentage) * data['SPY'].High if not isLong else (1 - self.limit_percentage) * data['SPY'].Low
            request = SubmitOrderRequest(orderType, self.security.Symbol.SecurityType, 'SPY', self.quantity, stopPrice, limitPrice, 0, 0.01, True, self.UtcTime, str(orderType))
            ticket = self.Transactions.AddOrder(request)
            self.tickets.append(ticket)
        elif len(self.tickets) > 0:
            ticket = self.tickets[-1]
            if self.Time.day > 8 and self.Time.day < 14:
                if len(ticket.UpdateRequests) == 0 and ticket.Status is not OrderStatus.Filled:
                    self.Log('TICKET:: {0}'.format(ticket))
                    updateOrderFields = UpdateOrderFields()
                    updateOrderFields.Quantity = ticket.Quantity + copysign(self.delta_quantity, self.quantity)
                    updateOrderFields.Tag = 'Change quantity: {0}'.format(self.Time.day)
                    ticket.Update(updateOrderFields)
            elif self.Time.day > 13 and self.Time.day < 20:
                if len(ticket.UpdateRequests) == 1 and ticket.Status is not OrderStatus.Filled:
                    self.Log('TICKET:: {0}'.format(ticket))
                    updateOrderFields = UpdateOrderFields()
                    updateOrderFields.LimitPrice = self.security.Price * (1 - copysign(self.limit_percentage_delta, ticket.Quantity))
                    updateOrderFields.StopPrice = self.security.Price * (1 + copysign(self.stop_percentage_delta, ticket.Quantity)) if ticket.OrderType != OrderType.TrailingStop else None
                    updateOrderFields.Tag = 'Change prices: {0}'.format(self.Time.day)
                    ticket.Update(updateOrderFields)
            elif len(ticket.UpdateRequests) == 2 and ticket.Status is not OrderStatus.Filled:
                self.Log('TICKET:: {0}'.format(ticket))
                ticket.Cancel('{0} and is still open!'.format(self.Time.day))
                self.Log('CANCELLED:: {0}'.format(ticket.CancelRequest))

    def OnOrderEvent(self, orderEvent):
        if False:
            i = 10
            return i + 15
        order = self.Transactions.GetOrderById(orderEvent.OrderId)
        ticket = self.Transactions.GetOrderTicket(orderEvent.OrderId)
        if order.Status == OrderStatus.Canceled and order.CanceledTime != orderEvent.UtcTime:
            raise ValueError('Expected canceled order CanceledTime to equal canceled order event time.')
        if (order.Status == OrderStatus.Filled or order.Status == OrderStatus.PartiallyFilled) and order.LastFillTime != orderEvent.UtcTime:
            raise ValueError('Expected filled order LastFillTime to equal fill order event time.')
        if len([ur for ur in ticket.UpdateRequests if ur.Response is not None and ur.Response.IsSuccess]) > 0 and order.CreatedTime != self.UtcTime and (order.LastUpdateTime is None):
            raise ValueError('Expected updated order LastUpdateTime to equal submitted update order event time')
        if orderEvent.Status == OrderStatus.Filled:
            self.Log('FILLED:: {0} FILL PRICE:: {1}'.format(self.Transactions.GetOrderById(orderEvent.OrderId), orderEvent.FillPrice))
        else:
            self.Log(orderEvent.ToString())
            self.Log('TICKET:: {0}'.format(ticket))