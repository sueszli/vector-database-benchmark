from AlgorithmImports import *

class OrderTicketDemoAlgorithm(QCAlgorithm):
    """In this algorithm we submit/update/cancel each order type"""

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        equity = self.AddEquity('SPY')
        self.spy = equity.Symbol
        self.__openMarketOnOpenOrders = []
        self.__openMarketOnCloseOrders = []
        self.__openLimitOrders = []
        self.__openStopMarketOrders = []
        self.__openStopLimitOrders = []
        self.__openTrailingStopOrders = []

    def OnData(self, data):
        if False:
            print('Hello World!')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        self.MarketOrders()
        self.LimitOrders()
        self.StopMarketOrders()
        self.StopLimitOrders()
        self.TrailingStopOrders()
        self.MarketOnOpenOrders()
        self.MarketOnCloseOrders()

    def MarketOrders(self):
        if False:
            print('Hello World!')
        " MarketOrders are the only orders that are processed synchronously by default, so\n        they'll fill by the next line of code. This behavior equally applies to live mode.\n        You can opt out of this behavior by specifying the 'asynchronous' parameter as True."
        if self.TimeIs(7, 9, 31):
            self.Log('Submitting MarketOrder')
            newTicket = self.MarketOrder(self.spy, 10, asynchronous=False)
            if newTicket.Status != OrderStatus.Filled:
                self.Log('Synchronous market order was not filled synchronously!')
                self.Quit()
            newTicket = self.MarketOrder(self.spy, 10, asynchronous=True)
            response = newTicket.Cancel('Attempt to cancel async order')
            if response.IsSuccess:
                self.Log('Successfully canceled async market order: {0}'.format(newTicket.OrderId))
            else:
                self.Log('Unable to cancel async market order: {0}'.format(response.ErrorCode))

    def LimitOrders(self):
        if False:
            return 10
        "LimitOrders are always processed asynchronously. Limit orders are used to\n        set 'good' entry points for an order. For example, you may wish to go\n        long a stock, but want a good price, so can place a LimitOrder to buy with\n        a limit price below the current market price. Likewise the opposite is True\n        when selling, you can place a LimitOrder to sell with a limit price above the\n        current market price to get a better sale price.\n        You can submit requests to update or cancel the LimitOrder at any time.\n        The 'LimitPrice' for an order can be retrieved from the ticket using the\n        OrderTicket.Get(OrderField) method, for example:\n        Code:\n            currentLimitPrice = orderTicket.Get(OrderField.LimitPrice)"
        if self.TimeIs(7, 12, 0):
            self.Log('Submitting LimitOrder')
            close = self.Securities[self.spy.Value].Close
            newTicket = self.LimitOrder(self.spy, 10, close * 0.999)
            self.__openLimitOrders.append(newTicket)
            newTicket = self.LimitOrder(self.spy, -10, close * 1.001)
            self.__openLimitOrders.append(newTicket)
        if len(self.__openLimitOrders) == 2:
            openOrders = self.__openLimitOrders
            longOrder = openOrders[0]
            shortOrder = openOrders[1]
            if self.CheckPairOrdersForFills(longOrder, shortOrder):
                self.__openLimitOrders = []
                return
            newLongLimit = longOrder.Get(OrderField.LimitPrice) + 0.01
            newShortLimit = shortOrder.Get(OrderField.LimitPrice) - 0.01
            self.Log('Updating limits - Long: {0:.2f} Short: {1:.2f}'.format(newLongLimit, newShortLimit))
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.LimitPrice = newLongLimit
            updateOrderFields.Tag = 'Update #{0}'.format(len(longOrder.UpdateRequests) + 1)
            longOrder.Update(updateOrderFields)
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.LimitPrice = newShortLimit
            updateOrderFields.Tag = 'Update #{0}'.format(len(shortOrder.UpdateRequests) + 1)
            shortOrder.Update(updateOrderFields)

    def StopMarketOrders(self):
        if False:
            return 10
        "StopMarketOrders work in the opposite way that limit orders do.\n        When placing a long trade, the stop price must be above current\n        market price. In this way it's a 'stop loss' for a short trade.\n        When placing a short trade, the stop price must be below current\n        market price. In this way it's a 'stop loss' for a long trade.\n        You can submit requests to update or cancel the StopMarketOrder at any time.\n        The 'StopPrice' for an order can be retrieved from the ticket using the\n        OrderTicket.Get(OrderField) method, for example:\n        Code:\n            currentStopPrice = orderTicket.Get(OrderField.StopPrice)"
        if self.TimeIs(7, 12 + 4, 0):
            self.Log('Submitting StopMarketOrder')
            close = self.Securities[self.spy.Value].Close
            newTicket = self.StopMarketOrder(self.spy, 10, close * 1.0025)
            self.__openStopMarketOrders.append(newTicket)
            newTicket = self.StopMarketOrder(self.spy, -10, close * 0.9975)
            self.__openStopMarketOrders.append(newTicket)
        if len(self.__openStopMarketOrders) == 2:
            longOrder = self.__openStopMarketOrders[0]
            shortOrder = self.__openStopMarketOrders[1]
            if self.CheckPairOrdersForFills(longOrder, shortOrder):
                self.__openStopMarketOrders = []
                return
            newLongStop = longOrder.Get(OrderField.StopPrice) - 0.01
            newShortStop = shortOrder.Get(OrderField.StopPrice) + 0.01
            self.Log('Updating stops - Long: {0:.2f} Short: {1:.2f}'.format(newLongStop, newShortStop))
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.StopPrice = newLongStop
            updateOrderFields.Tag = 'Update #{0}'.format(len(longOrder.UpdateRequests) + 1)
            longOrder.Update(updateOrderFields)
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.StopPrice = newShortStop
            updateOrderFields.Tag = 'Update #{0}'.format(len(shortOrder.UpdateRequests) + 1)
            shortOrder.Update(updateOrderFields)
            self.Log('Updated price - Long: {0} Short: {1}'.format(longOrder.Get(OrderField.StopPrice), shortOrder.Get(OrderField.StopPrice)))

    def StopLimitOrders(self):
        if False:
            while True:
                i = 10
        "StopLimitOrders work as a combined stop and limit order. First, the\n        price must pass the stop price in the same way a StopMarketOrder works,\n        but then we're also guaranteed a fill price at least as good as the\n        limit price. This order type can be beneficial in gap down scenarios\n        where a StopMarketOrder would have triggered and given the not as beneficial\n        gapped down price, whereas the StopLimitOrder could protect you from\n        getting the gapped down price through prudent placement of the limit price.\n        You can submit requests to update or cancel the StopLimitOrder at any time.\n        The 'StopPrice' or 'LimitPrice' for an order can be retrieved from the ticket\n        using the OrderTicket.Get(OrderField) method, for example:\n        Code:\n            currentStopPrice = orderTicket.Get(OrderField.StopPrice)\n            currentLimitPrice = orderTicket.Get(OrderField.LimitPrice)"
        if self.TimeIs(8, 12, 1):
            self.Log('Submitting StopLimitOrder')
            close = self.Securities[self.spy.Value].Close
            newTicket = self.StopLimitOrder(self.spy, 10, close * 1.001, close - 0.03)
            self.__openStopLimitOrders.append(newTicket)
            newTicket = self.StopLimitOrder(self.spy, -10, close * 0.999, close + 0.03)
            self.__openStopLimitOrders.append(newTicket)
        if len(self.__openStopLimitOrders) == 2:
            longOrder = self.__openStopLimitOrders[0]
            shortOrder = self.__openStopLimitOrders[1]
            if self.CheckPairOrdersForFills(longOrder, shortOrder):
                self.__openStopLimitOrders = []
                return
            newLongStop = longOrder.Get(OrderField.StopPrice) - 0.01
            newLongLimit = longOrder.Get(OrderField.LimitPrice) + 0.01
            newShortStop = shortOrder.Get(OrderField.StopPrice) + 0.01
            newShortLimit = shortOrder.Get(OrderField.LimitPrice) - 0.01
            self.Log('Updating stops  - Long: {0:.2f} Short: {1:.2f}'.format(newLongStop, newShortStop))
            self.Log('Updating limits - Long: {0:.2f}  Short: {1:.2f}'.format(newLongLimit, newShortLimit))
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.StopPrice = newLongStop
            updateOrderFields.LimitPrice = newLongLimit
            updateOrderFields.Tag = 'Update #{0}'.format(len(longOrder.UpdateRequests) + 1)
            longOrder.Update(updateOrderFields)
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.StopPrice = newShortStop
            updateOrderFields.LimitPrice = newShortLimit
            updateOrderFields.Tag = 'Update #{0}'.format(len(shortOrder.UpdateRequests) + 1)
            shortOrder.Update(updateOrderFields)

    def TrailingStopOrders(self):
        if False:
            while True:
                i = 10
        'TrailingStopOrders work the same way as StopMarketOrders, except\n        their stop price is adjusted to a certain amount, keeping it a certain\n        fixed distance from/to the market price, depending on the order direction,\n        which allows to preserve profits and protecting against losses.\n        The stop price can be accessed just as with StopMarketOrders, and\n        the trailing amount can be accessed with the OrderTicket.Get(OrderField), for example:\n        Code:\n            currentTrailingAmount = orderTicket.Get(OrderField.StopPrice)\n            trailingAsPercentage = orderTicket.Get[bool](OrderField.TrailingAsPercentage)'
        if self.TimeIs(7, 12, 0):
            self.Log('Submitting TrailingStopOrder')
            close = self.Securities[self.spy.Value].Close
            stopPrice = close * 1.0025
            newTicket = self.TrailingStopOrder(self.spy, 10, stopPrice, trailingAmount=0.0025, trailingAsPercentage=True)
            self.__openTrailingStopOrders.append(newTicket)
            stopPrice = close * 0.9975
            newTicket = self.TrailingStopOrder(self.spy, -10, stopPrice, trailingAmount=0.0025, trailingAsPercentage=True)
            self.__openTrailingStopOrders.append(newTicket)
        elif len(self.__openTrailingStopOrders) == 2:
            longOrder = self.__openTrailingStopOrders[0]
            shortOrder = self.__openTrailingStopOrders[1]
            if self.CheckPairOrdersForFills(longOrder, shortOrder):
                self.__openTrailingStopOrders = []
                return
            if (self.UtcTime - longOrder.Time).total_seconds() / 60 % 5 != 0:
                return
            longTrailingPercentage = longOrder.Get(OrderField.TrailingAmount)
            newLongTrailingPercentage = max(longTrailingPercentage - 0.0001, 0.0001)
            shortTrailingPercentage = shortOrder.Get(OrderField.TrailingAmount)
            newShortTrailingPercentage = max(shortTrailingPercentage - 0.0001, 0.0001)
            self.Log('Updating trailing percentages - Long: {0:.3f} Short: {1:.3f}'.format(newLongTrailingPercentage, newShortTrailingPercentage))
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.TrailingAmount = newLongTrailingPercentage
            updateOrderFields.Tag = 'Update #{0}'.format(len(longOrder.UpdateRequests) + 1)
            longOrder.Update(updateOrderFields)
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.TrailingAmount = newShortTrailingPercentage
            updateOrderFields.Tag = 'Update #{0}'.format(len(shortOrder.UpdateRequests) + 1)
            shortOrder.Update(updateOrderFields)

    def MarketOnCloseOrders(self):
        if False:
            print('Hello World!')
        "MarketOnCloseOrders are always executed at the next market's closing price.\n        The only properties that can be updated are the quantity and order tag properties."
        if self.TimeIs(9, 12, 0):
            self.Log('Submitting MarketOnCloseOrder')
            qty = self.Portfolio[self.spy.Value].Quantity
            qty = 100 if qty == 0 else 2 * qty
            newTicket = self.MarketOnCloseOrder(self.spy, qty)
            self.__openMarketOnCloseOrders.append(newTicket)
        if len(self.__openMarketOnCloseOrders) == 1 and self.Time.minute == 59:
            ticket = self.__openMarketOnCloseOrders[0]
            if ticket.Status == OrderStatus.Filled:
                self.__openMarketOnCloseOrders = []
                return
            quantity = ticket.Quantity + 1
            self.Log('Updating quantity  - New Quantity: {0}'.format(quantity))
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.Quantity = quantity
            updateOrderFields.Tag = 'Update #{0}'.format(len(ticket.UpdateRequests) + 1)
            ticket.Update(updateOrderFields)
        if self.TimeIs(self.EndDate.day, 12 + 3, 45):
            self.Log('Submitting MarketOnCloseOrder to liquidate end of algorithm')
            self.MarketOnCloseOrder(self.spy, -self.Portfolio[self.spy.Value].Quantity, 'Liquidate end of algorithm')

    def MarketOnOpenOrders(self):
        if False:
            while True:
                i = 10
        "MarketOnOpenOrders are always executed at the next\n        market's opening price. The only properties that can\n        be updated are the quantity and order tag properties."
        if self.TimeIs(8, 12 + 2, 0):
            self.Log('Submitting MarketOnOpenOrder')
            newTicket = self.MarketOnOpenOrder(self.spy, 50)
            self.__openMarketOnOpenOrders.append(newTicket)
        if len(self.__openMarketOnOpenOrders) == 1 and self.Time.minute == 59:
            ticket = self.__openMarketOnOpenOrders[0]
            if ticket.Status == OrderStatus.Filled:
                self.__openMarketOnOpenOrders = []
                return
            quantity = ticket.Quantity + 1
            self.Log('Updating quantity  - New Quantity: {0}'.format(quantity))
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.Quantity = quantity
            updateOrderFields.Tag = 'Update #{0}'.format(len(ticket.UpdateRequests) + 1)
            ticket.Update(updateOrderFields)

    def OnOrderEvent(self, orderEvent):
        if False:
            print('Hello World!')
        order = self.Transactions.GetOrderById(orderEvent.OrderId)
        self.Log('{0}: {1}: {2}'.format(self.Time, order.Type, orderEvent))
        if orderEvent.Quantity == 0:
            raise Exception('OrderEvent quantity is Not expected to be 0, it should hold the current order Quantity')
        if orderEvent.Quantity != order.Quantity:
            raise Exception('OrderEvent quantity should hold the current order Quantity')
        if type(order) is LimitOrder and orderEvent.LimitPrice == 0 or (type(order) is StopLimitOrder and orderEvent.LimitPrice == 0):
            raise Exception('OrderEvent LimitPrice is Not expected to be 0 for LimitOrder and StopLimitOrder')
        if type(order) is StopMarketOrder and orderEvent.StopPrice == 0:
            raise Exception('OrderEvent StopPrice is Not expected to be 0 for StopMarketOrder')
        if orderEvent.Ticket is None:
            raise Exception('OrderEvent Ticket was not set')
        if orderEvent.OrderId != orderEvent.Ticket.OrderId:
            raise Exception('OrderEvent.OrderId and orderEvent.Ticket.OrderId do not match')

    def CheckPairOrdersForFills(self, longOrder, shortOrder):
        if False:
            return 10
        if longOrder.Status == OrderStatus.Filled:
            self.Log('{0}: Cancelling short order, long order is filled.'.format(shortOrder.OrderType))
            shortOrder.Cancel('Long filled.')
            return True
        if shortOrder.Status == OrderStatus.Filled:
            self.Log('{0}: Cancelling long order, short order is filled.'.format(longOrder.OrderType))
            longOrder.Cancel('Short filled')
            return True
        return False

    def TimeIs(self, day, hour, minute):
        if False:
            return 10
        return self.Time.day == day and self.Time.hour == hour and (self.Time.minute == minute)

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        basicOrderTicketFilter = lambda x: x.Symbol == self.spy
        filledOrders = self.Transactions.GetOrders(lambda x: x.Status == OrderStatus.Filled)
        orderTickets = self.Transactions.GetOrderTickets(basicOrderTicketFilter)
        openOrders = self.Transactions.GetOpenOrders(lambda x: x.Symbol == self.spy)
        openOrderTickets = self.Transactions.GetOpenOrderTickets(basicOrderTicketFilter)
        remainingOpenOrders = self.Transactions.GetOpenOrdersRemainingQuantity(basicOrderTicketFilter)
        filledOrdersSize = sum((1 for order in filledOrders))
        orderTicketsSize = sum((1 for ticket in orderTickets))
        openOrderTicketsSize = sum((1 for ticket in openOrderTickets))
        assert filledOrdersSize == 9 and orderTicketsSize == 12, 'There were expected 9 filled orders and 12 order tickets'
        assert not (len(openOrders) or openOrderTicketsSize), 'No open orders or tickets were expected'
        assert not remainingOpenOrders, 'No remaining quantity to be filled from open orders was expected'
        spyOpenOrders = self.Transactions.GetOpenOrders(self.spy)
        spyOpenOrderTickets = self.Transactions.GetOpenOrderTickets(self.spy)
        spyOpenOrderTicketsSize = sum((1 for tickets in spyOpenOrderTickets))
        spyOpenOrdersRemainingQuantity = self.Transactions.GetOpenOrdersRemainingQuantity(self.spy)
        assert not (len(spyOpenOrders) or spyOpenOrderTicketsSize), 'No open orders or tickets were expected'
        assert not spyOpenOrdersRemainingQuantity, 'No remaining quantity to be filled from open orders was expected'
        defaultOrders = self.Transactions.GetOrders()
        defaultOrderTickets = self.Transactions.GetOrderTickets()
        defaultOpenOrders = self.Transactions.GetOpenOrders()
        defaultOpenOrderTickets = self.Transactions.GetOpenOrderTickets()
        defaultOpenOrdersRemaining = self.Transactions.GetOpenOrdersRemainingQuantity()
        defaultOrdersSize = sum((1 for order in defaultOrders))
        defaultOrderTicketsSize = sum((1 for ticket in defaultOrderTickets))
        defaultOpenOrderTicketsSize = sum((1 for ticket in defaultOpenOrderTickets))
        assert defaultOrdersSize == 12 and defaultOrderTicketsSize == 12, 'There were expected 12 orders and 12 order tickets'
        assert not (len(defaultOpenOrders) or defaultOpenOrderTicketsSize), 'No open orders or tickets were expected'
        assert not defaultOpenOrdersRemaining, 'No remaining quantity to be filled from open orders was expected'