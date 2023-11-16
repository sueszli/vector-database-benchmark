# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from AlgorithmImports import *

### <summary>
### In this algorithm we submit/update/cancel each order type
### </summary>
### <meta name="tag" content="trading and orders" />
### <meta name="tag" content="placing orders" />
### <meta name="tag" content="managing orders" />
### <meta name="tag" content="order tickets" />
### <meta name="tag" content="updating orders" />
class OrderTicketDemoAlgorithm(QCAlgorithm):
    '''In this algorithm we submit/update/cancel each order type'''
    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        self.SetStartDate(2013,10,7)   #Set Start Date
        self.SetEndDate(2013,10,11)    #Set End Date
        self.SetCash(100000)           #Set Strategy Cash
        # Find more symbols here: http://quantconnect.com/data
        equity = self.AddEquity("SPY")
        self.spy = equity.Symbol

        self.__openMarketOnOpenOrders = []
        self.__openMarketOnCloseOrders = []
        self.__openLimitOrders = []
        self.__openStopMarketOrders = []
        self.__openStopLimitOrders = []
        self.__openTrailingStopOrders = []


    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'''
        # MARKET ORDERS
        self.MarketOrders()

        # LIMIT ORDERS
        self.LimitOrders()

        # STOP MARKET ORDERS
        self.StopMarketOrders()

        # STOP LIMIT ORDERS
        self.StopLimitOrders()

        # TRAILING STOP ORDERS
        self.TrailingStopOrders()

        # MARKET ON OPEN ORDERS
        self.MarketOnOpenOrders()

        # MARKET ON CLOSE ORDERS
        self.MarketOnCloseOrders()


    def MarketOrders(self):
        ''' MarketOrders are the only orders that are processed synchronously by default, so
        they'll fill by the next line of code. This behavior equally applies to live mode.
        You can opt out of this behavior by specifying the 'asynchronous' parameter as True.'''
        if self.TimeIs(7, 9, 31):
            self.Log("Submitting MarketOrder")

            # submit a market order to buy 10 shares, this function returns an OrderTicket object
            # we submit the order with asynchronous = False, so it block until it is filled
            newTicket = self.MarketOrder(self.spy, 10, asynchronous = False)
            if newTicket.Status != OrderStatus.Filled:
                self.Log("Synchronous market order was not filled synchronously!")
                self.Quit()

            # we can also submit the ticket asynchronously. In a backtest, we'll still perform the fill
            # before the next time events for your algorithm. here we'll submit the order asynchronously
            # and try to cancel it, sometimes it will, sometimes it will be filled first.
            newTicket = self.MarketOrder(self.spy, 10, asynchronous = True)
            response = newTicket.Cancel("Attempt to cancel async order")
            if response.IsSuccess:
                self.Log("Successfully canceled async market order: {0}".format(newTicket.OrderId))
            else:
                self.Log("Unable to cancel async market order: {0}".format(response.ErrorCode))


    def LimitOrders(self):
        '''LimitOrders are always processed asynchronously. Limit orders are used to
        set 'good' entry points for an order. For example, you may wish to go
        long a stock, but want a good price, so can place a LimitOrder to buy with
        a limit price below the current market price. Likewise the opposite is True
        when selling, you can place a LimitOrder to sell with a limit price above the
        current market price to get a better sale price.
        You can submit requests to update or cancel the LimitOrder at any time.
        The 'LimitPrice' for an order can be retrieved from the ticket using the
        OrderTicket.Get(OrderField) method, for example:
        Code:
            currentLimitPrice = orderTicket.Get(OrderField.LimitPrice)'''
        if self.TimeIs(7, 12, 0):
            self.Log("Submitting LimitOrder")

            # submit a limit order to buy 10 shares at .1% below the bar's close
            close = self.Securities[self.spy.Value].Close
            newTicket = self.LimitOrder(self.spy, 10, close * .999)
            self.__openLimitOrders.append(newTicket)

            # submit another limit order to sell 10 shares at .1% above the bar's close
            newTicket = self.LimitOrder(self.spy, -10, close * 1.001)
            self.__openLimitOrders.append(newTicket)

        # when we submitted new limit orders we placed them into this list,
        # so while there's two entries they're still open and need processing
        if len(self.__openLimitOrders) == 2:
            openOrders = self.__openLimitOrders

            # check if either is filled and cancel the other
            longOrder = openOrders[0]
            shortOrder = openOrders[1]
            if self.CheckPairOrdersForFills(longOrder, shortOrder):
                self.__openLimitOrders = []
                return

            # if neither order has filled, bring in the limits by a penny
            newLongLimit = longOrder.Get(OrderField.LimitPrice) + 0.01
            newShortLimit = shortOrder.Get(OrderField.LimitPrice) - 0.01
            self.Log("Updating limits - Long: {0:.2f} Short: {1:.2f}".format(newLongLimit, newShortLimit))

            updateOrderFields = UpdateOrderFields()
            updateOrderFields.LimitPrice = newLongLimit
            updateOrderFields.Tag = "Update #{0}".format(len(longOrder.UpdateRequests) + 1)
            longOrder.Update(updateOrderFields)

            updateOrderFields = UpdateOrderFields()
            updateOrderFields.LimitPrice = newShortLimit
            updateOrderFields.Tag = "Update #{0}".format(len(shortOrder.UpdateRequests) + 1)
            shortOrder.Update(updateOrderFields)


    def StopMarketOrders(self):
        '''StopMarketOrders work in the opposite way that limit orders do.
        When placing a long trade, the stop price must be above current
        market price. In this way it's a 'stop loss' for a short trade.
        When placing a short trade, the stop price must be below current
        market price. In this way it's a 'stop loss' for a long trade.
        You can submit requests to update or cancel the StopMarketOrder at any time.
        The 'StopPrice' for an order can be retrieved from the ticket using the
        OrderTicket.Get(OrderField) method, for example:
        Code:
            currentStopPrice = orderTicket.Get(OrderField.StopPrice)'''
        if self.TimeIs(7, 12 + 4, 0):
            self.Log("Submitting StopMarketOrder")

            # a long stop is triggered when the price rises above the value
            # so we'll set a long stop .25% above the current bar's close
            close = self.Securities[self.spy.Value].Close
            newTicket = self.StopMarketOrder(self.spy, 10, close * 1.0025)
            self.__openStopMarketOrders.append(newTicket)

            # a short stop is triggered when the price falls below the value
            # so we'll set a short stop .25% below the current bar's close
            newTicket = self.StopMarketOrder(self.spy, -10, close * .9975)
            self.__openStopMarketOrders.append(newTicket)

        # when we submitted new stop market orders we placed them into this list,
        # so while there's two entries they're still open and need processing
        if len(self.__openStopMarketOrders) == 2:
            # check if either is filled and cancel the other
            longOrder = self.__openStopMarketOrders[0]
            shortOrder = self.__openStopMarketOrders[1]
            if self.CheckPairOrdersForFills(longOrder, shortOrder):
                self.__openStopMarketOrders = []
                return

            # if neither order has filled, bring in the stops by a penny
            newLongStop = longOrder.Get(OrderField.StopPrice) - 0.01
            newShortStop = shortOrder.Get(OrderField.StopPrice) + 0.01
            self.Log("Updating stops - Long: {0:.2f} Short: {1:.2f}".format(newLongStop, newShortStop))

            updateOrderFields = UpdateOrderFields()
            updateOrderFields.StopPrice = newLongStop
            updateOrderFields.Tag = "Update #{0}".format(len(longOrder.UpdateRequests) + 1)
            longOrder.Update(updateOrderFields)

            updateOrderFields = UpdateOrderFields()
            updateOrderFields.StopPrice = newShortStop
            updateOrderFields.Tag = "Update #{0}".format(len(shortOrder.UpdateRequests) + 1)
            shortOrder.Update(updateOrderFields)
            self.Log("Updated price - Long: {0} Short: {1}".format(longOrder.Get(OrderField.StopPrice), shortOrder.Get(OrderField.StopPrice)))


    def StopLimitOrders(self):
        '''StopLimitOrders work as a combined stop and limit order. First, the
        price must pass the stop price in the same way a StopMarketOrder works,
        but then we're also guaranteed a fill price at least as good as the
        limit price. This order type can be beneficial in gap down scenarios
        where a StopMarketOrder would have triggered and given the not as beneficial
        gapped down price, whereas the StopLimitOrder could protect you from
        getting the gapped down price through prudent placement of the limit price.
        You can submit requests to update or cancel the StopLimitOrder at any time.
        The 'StopPrice' or 'LimitPrice' for an order can be retrieved from the ticket
        using the OrderTicket.Get(OrderField) method, for example:
        Code:
            currentStopPrice = orderTicket.Get(OrderField.StopPrice)
            currentLimitPrice = orderTicket.Get(OrderField.LimitPrice)'''
        if self.TimeIs(8, 12, 1):
            self.Log("Submitting StopLimitOrder")

            # a long stop is triggered when the price rises above the
            # value so we'll set a long stop .25% above the current bar's
            # close now we'll also be setting a limit, this means we are
            # guaranteed to get at least the limit price for our fills,
            # so make the limit price a little higher than the stop price

            close = self.Securities[self.spy.Value].Close
            newTicket = self.StopLimitOrder(self.spy, 10, close * 1.001, close - 0.03)
            self.__openStopLimitOrders.append(newTicket)

            # a short stop is triggered when the price falls below the
            # value so we'll set a short stop .25% below the current bar's
            # close now we'll also be setting a limit, this means we are
            # guaranteed to get at least the limit price for our fills,
            # so make the limit price a little softer than the stop price

            newTicket = self.StopLimitOrder(self.spy, -10, close * .999, close + 0.03)
            self.__openStopLimitOrders.append(newTicket)

        # when we submitted new stop limit orders we placed them into this list,
        # so while there's two entries they're still open and need processing
        if len(self.__openStopLimitOrders) == 2:
            longOrder = self.__openStopLimitOrders[0]
            shortOrder = self.__openStopLimitOrders[1]
            if self.CheckPairOrdersForFills(longOrder, shortOrder):
                self.__openStopLimitOrders = []
                return

            # if neither order has filled, bring in the stops/limits in by a penny

            newLongStop = longOrder.Get(OrderField.StopPrice) - 0.01
            newLongLimit = longOrder.Get(OrderField.LimitPrice) + 0.01
            newShortStop = shortOrder.Get(OrderField.StopPrice) + 0.01
            newShortLimit = shortOrder.Get(OrderField.LimitPrice) - 0.01
            self.Log("Updating stops  - Long: {0:.2f} Short: {1:.2f}".format(newLongStop, newShortStop))
            self.Log("Updating limits - Long: {0:.2f}  Short: {1:.2f}".format(newLongLimit, newShortLimit))

            updateOrderFields = UpdateOrderFields()
            updateOrderFields.StopPrice = newLongStop
            updateOrderFields.LimitPrice = newLongLimit
            updateOrderFields.Tag = "Update #{0}".format(len(longOrder.UpdateRequests) + 1)
            longOrder.Update(updateOrderFields)

            updateOrderFields = UpdateOrderFields()
            updateOrderFields.StopPrice = newShortStop
            updateOrderFields.LimitPrice = newShortLimit
            updateOrderFields.Tag = "Update #{0}".format(len(shortOrder.UpdateRequests) + 1)
            shortOrder.Update(updateOrderFields)


    def TrailingStopOrders(self):
        '''TrailingStopOrders work the same way as StopMarketOrders, except
        their stop price is adjusted to a certain amount, keeping it a certain
        fixed distance from/to the market price, depending on the order direction,
        which allows to preserve profits and protecting against losses.
        The stop price can be accessed just as with StopMarketOrders, and
        the trailing amount can be accessed with the OrderTicket.Get(OrderField), for example:
        Code:
            currentTrailingAmount = orderTicket.Get(OrderField.StopPrice)
            trailingAsPercentage = orderTicket.Get[bool](OrderField.TrailingAsPercentage)'''
        if self.TimeIs(7, 12, 0):
            self.Log("Submitting TrailingStopOrder")

            # a long stop is triggered when the price rises above the
            # value so we'll set a long stop .25% above the current bar's

            close = self.Securities[self.spy.Value].Close
            stopPrice = close * 1.0025
            newTicket = self.TrailingStopOrder(self.spy, 10, stopPrice, trailingAmount=0.0025, trailingAsPercentage=True)
            self.__openTrailingStopOrders.append(newTicket)

            # a short stop is triggered when the price falls below the
            # value so we'll set a short stop .25% below the current bar's

            stopPrice = close * .9975
            newTicket = self.TrailingStopOrder(self.spy, -10, stopPrice, trailingAmount=0.0025, trailingAsPercentage=True)
            self.__openTrailingStopOrders.append(newTicket)

        # when we submitted new stop market orders we placed them into this list,
        # so while there's two entries they're still open and need processing
        elif len(self.__openTrailingStopOrders) == 2:
            longOrder = self.__openTrailingStopOrders[0]
            shortOrder = self.__openTrailingStopOrders[1]
            if self.CheckPairOrdersForFills(longOrder, shortOrder):
                self.__openTrailingStopOrders = []
                return

            # if neither order has filled in the last 5 minutes, bring in the trailing percentage by 0.01%
            if ((self.UtcTime - longOrder.Time).total_seconds() / 60) % 5 != 0:
                return

            longTrailingPercentage = longOrder.Get(OrderField.TrailingAmount)
            newLongTrailingPercentage = max(longTrailingPercentage - 0.0001, 0.0001)
            shortTrailingPercentage = shortOrder.Get(OrderField.TrailingAmount)
            newShortTrailingPercentage = max(shortTrailingPercentage - 0.0001, 0.0001)
            self.Log("Updating trailing percentages - Long: {0:.3f} Short: {1:.3f}".format(newLongTrailingPercentage, newShortTrailingPercentage))

            updateOrderFields = UpdateOrderFields()
            # we could change the quantity, but need to specify it
            #Quantity =
            updateOrderFields.TrailingAmount = newLongTrailingPercentage
            updateOrderFields.Tag = "Update #{0}".format(len(longOrder.UpdateRequests) + 1)
            longOrder.Update(updateOrderFields)

            updateOrderFields = UpdateOrderFields()
            updateOrderFields.TrailingAmount = newShortTrailingPercentage
            updateOrderFields.Tag = "Update #{0}".format(len(shortOrder.UpdateRequests) + 1)
            shortOrder.Update(updateOrderFields)


    def MarketOnCloseOrders(self):
        '''MarketOnCloseOrders are always executed at the next market's closing price.
        The only properties that can be updated are the quantity and order tag properties.'''
        if self.TimeIs(9, 12, 0):
            self.Log("Submitting MarketOnCloseOrder")

            # open a new position or triple our existing position
            qty = self.Portfolio[self.spy.Value].Quantity
            qty = 100 if qty == 0 else 2*qty

            newTicket = self.MarketOnCloseOrder(self.spy, qty)
            self.__openMarketOnCloseOrders.append(newTicket)

        if len(self.__openMarketOnCloseOrders) == 1 and self.Time.minute == 59:
            ticket = self.__openMarketOnCloseOrders[0]
            # check for fills
            if ticket.Status == OrderStatus.Filled:
                self.__openMarketOnCloseOrders = []
                return

            quantity = ticket.Quantity + 1
            self.Log("Updating quantity  - New Quantity: {0}".format(quantity))

            # we can update the quantity and tag
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.Quantity = quantity
            updateOrderFields.Tag = "Update #{0}".format(len(ticket.UpdateRequests) + 1)
            ticket.Update(updateOrderFields)

        if self.TimeIs(self.EndDate.day, 12 + 3, 45):
            self.Log("Submitting MarketOnCloseOrder to liquidate end of algorithm")
            self.MarketOnCloseOrder(self.spy, -self.Portfolio[self.spy.Value].Quantity, "Liquidate end of algorithm")


    def MarketOnOpenOrders(self):
        '''MarketOnOpenOrders are always executed at the next
        market's opening price. The only properties that can
        be updated are the quantity and order tag properties.'''
        if self.TimeIs(8, 12 + 2, 0):
            self.Log("Submitting MarketOnOpenOrder")

            # its EOD, let's submit a market on open order to short even more!
            newTicket = self.MarketOnOpenOrder(self.spy, 50)
            self.__openMarketOnOpenOrders.append(newTicket)

        if len(self.__openMarketOnOpenOrders) == 1 and self.Time.minute == 59:
            ticket = self.__openMarketOnOpenOrders[0]

            # check for fills
            if ticket.Status == OrderStatus.Filled:
                self.__openMarketOnOpenOrders = []
                return

            quantity = ticket.Quantity + 1
            self.Log("Updating quantity  - New Quantity: {0}".format(quantity))

            # we can update the quantity and tag
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.Quantity = quantity
            updateOrderFields.Tag = "Update #{0}".format(len(ticket.UpdateRequests) + 1)
            ticket.Update(updateOrderFields)


    def OnOrderEvent(self, orderEvent):
        order = self.Transactions.GetOrderById(orderEvent.OrderId)
        self.Log("{0}: {1}: {2}".format(self.Time, order.Type, orderEvent))

        if orderEvent.Quantity == 0:
            raise Exception("OrderEvent quantity is Not expected to be 0, it should hold the current order Quantity")

        if orderEvent.Quantity != order.Quantity:
            raise Exception("OrderEvent quantity should hold the current order Quantity")

        if (type(order) is LimitOrder and orderEvent.LimitPrice == 0 or
            type(order) is StopLimitOrder and orderEvent.LimitPrice == 0):
            raise Exception("OrderEvent LimitPrice is Not expected to be 0 for LimitOrder and StopLimitOrder")

        if type(order) is StopMarketOrder and orderEvent.StopPrice == 0:
            raise Exception("OrderEvent StopPrice is Not expected to be 0 for StopMarketOrder")

        # We can access the order ticket from the order event
        if orderEvent.Ticket is None:
            raise Exception("OrderEvent Ticket was not set")
        if orderEvent.OrderId != orderEvent.Ticket.OrderId:
            raise Exception("OrderEvent.OrderId and orderEvent.Ticket.OrderId do not match")

    def CheckPairOrdersForFills(self, longOrder, shortOrder):
        if longOrder.Status == OrderStatus.Filled:
            self.Log("{0}: Cancelling short order, long order is filled.".format(shortOrder.OrderType))
            shortOrder.Cancel("Long filled.")
            return True

        if shortOrder.Status == OrderStatus.Filled:
            self.Log("{0}: Cancelling long order, short order is filled.".format(longOrder.OrderType))
            longOrder.Cancel("Short filled")
            return True

        return False


    def TimeIs(self, day, hour, minute):
        return self.Time.day == day and self.Time.hour == hour and self.Time.minute == minute

    def OnEndOfAlgorithm(self):
        basicOrderTicketFilter = lambda x: x.Symbol == self.spy

        filledOrders = self.Transactions.GetOrders(lambda x: x.Status == OrderStatus.Filled)
        orderTickets = self.Transactions.GetOrderTickets(basicOrderTicketFilter)
        openOrders = self.Transactions.GetOpenOrders(lambda x: x.Symbol == self.spy)
        openOrderTickets = self.Transactions.GetOpenOrderTickets(basicOrderTicketFilter)
        remainingOpenOrders = self.Transactions.GetOpenOrdersRemainingQuantity(basicOrderTicketFilter)

        # The type returned by self.Transactions.GetOrders() is iterable and not a list
        # that's why we use sum() to get the size of the iterable object type
        filledOrdersSize = sum(1 for order in filledOrders)
        orderTicketsSize = sum(1 for ticket in orderTickets)
        openOrderTicketsSize = sum(1 for ticket in openOrderTickets)

        assert(filledOrdersSize == 9 and orderTicketsSize == 12), "There were expected 9 filled orders and 12 order tickets"
        assert(not (len(openOrders) or openOrderTicketsSize)), "No open orders or tickets were expected"
        assert(not remainingOpenOrders), "No remaining quantity to be filled from open orders was expected"

        spyOpenOrders = self.Transactions.GetOpenOrders(self.spy)
        spyOpenOrderTickets = self.Transactions.GetOpenOrderTickets(self.spy)
        spyOpenOrderTicketsSize = sum(1 for tickets in spyOpenOrderTickets)
        spyOpenOrdersRemainingQuantity = self.Transactions.GetOpenOrdersRemainingQuantity(self.spy)

        assert(not (len(spyOpenOrders) or spyOpenOrderTicketsSize)), "No open orders or tickets were expected"
        assert(not spyOpenOrdersRemainingQuantity), "No remaining quantity to be filled from open orders was expected"

        defaultOrders = self.Transactions.GetOrders()
        defaultOrderTickets = self.Transactions.GetOrderTickets()
        defaultOpenOrders = self.Transactions.GetOpenOrders()
        defaultOpenOrderTickets = self.Transactions.GetOpenOrderTickets()
        defaultOpenOrdersRemaining = self.Transactions.GetOpenOrdersRemainingQuantity()

        defaultOrdersSize = sum(1 for order in defaultOrders)
        defaultOrderTicketsSize = sum(1 for ticket in defaultOrderTickets)
        defaultOpenOrderTicketsSize = sum(1 for ticket in defaultOpenOrderTickets)

        assert(defaultOrdersSize == 12 and defaultOrderTicketsSize == 12), "There were expected 12 orders and 12 order tickets"
        assert(not (len(defaultOpenOrders) or defaultOpenOrderTicketsSize)), "No open orders or tickets were expected"
        assert(not defaultOpenOrdersRemaining), "No remaining quantity to be filled from open orders was expected"
