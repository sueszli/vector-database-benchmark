import itertools
from AlgorithmImports import *

class ComboOrderTicketDemoAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)
        self.SetCash(100000)
        equity = self.AddEquity('GOOG', leverage=4, fillForward=True)
        option = self.AddOption(equity.Symbol, fillForward=True)
        self._optionSymbol = option.Symbol
        option.SetFilter(lambda u: u.Strikes(-2, +2).Expiration(0, 180))
        self._openMarketOrders = []
        self._openLegLimitOrders = []
        self._openLimitOrders = []
        self._orderLegs = None

    def OnData(self, data: Slice):
        if False:
            return 10
        if self._orderLegs is None:
            if self.IsMarketOpen(self._optionSymbol):
                chain = data.OptionChains.GetValue(self._optionSymbol)
                if chain is not None:
                    callContracts = [contract for contract in chain if contract.Right == OptionRight.Call]
                    callContracts = [(key, list(group)) for (key, group) in itertools.groupby(callContracts, key=lambda x: x.Expiry)]
                    callContracts.sort(key=lambda x: x[0])
                    callContracts = callContracts[0][1]
                    callContracts.sort(key=lambda x: x.Strike)
                    if len(callContracts) < 3:
                        return
                    quantities = [1, -2, 1]
                    self._orderLegs = []
                    for (i, contract) in enumerate(callContracts[:3]):
                        leg = Leg.Create(contract.Symbol, quantities[i])
                        self._orderLegs.append(leg)
        else:
            self.ComboMarketOrders()
            self.ComboLimitOrders()
            self.ComboLegLimitOrders()

    def ComboMarketOrders(self):
        if False:
            print('Hello World!')
        if len(self._openMarketOrders) != 0 or self._orderLegs is None:
            return
        self.Log('Submitting combo market orders')
        tickets = self.ComboMarketOrder(self._orderLegs, 2, asynchronous=False)
        self._openMarketOrders.extend(tickets)
        tickets = self.ComboMarketOrder(self._orderLegs, 2, asynchronous=True)
        self._openMarketOrders.extend(tickets)
        for ticket in tickets:
            response = ticket.Cancel('Attempt to cancel combo market order')
            if response.IsSuccess:
                raise Exception('Combo market orders should fill instantly, they should not be cancelable in backtest mode: ' + response.OrderId)

    def ComboLimitOrders(self):
        if False:
            i = 10
            return i + 15
        if len(self._openLimitOrders) == 0:
            self.Log('Submitting ComboLimitOrder')
            currentPrice = sum([leg.Quantity * self.Securities[leg.Symbol].Close for leg in self._orderLegs])
            tickets = self.ComboLimitOrder(self._orderLegs, 2, currentPrice + 1.5)
            self._openLimitOrders.extend(tickets)
            tickets = self.ComboLimitOrder(self._orderLegs, -2, currentPrice + 3)
            self._openLimitOrders.extend(tickets)
        else:
            combo1 = self._openLimitOrders[:len(self._orderLegs)]
            combo2 = self._openLimitOrders[-len(self._orderLegs):]
            if self.CheckGroupOrdersForFills(combo1, combo2):
                return
            ticket = combo1[0]
            newLimit = round(ticket.Get(OrderField.LimitPrice) + 0.01, 2)
            self.Debug(f'Updating limits - Combo 1 {ticket.OrderId}: {newLimit:.2f}')
            fields = UpdateOrderFields()
            fields.LimitPrice = newLimit
            fields.Tag = f'Update #{len(ticket.UpdateRequests) + 1}'
            ticket.Update(fields)
            ticket = combo2[0]
            newLimit = round(ticket.Get(OrderField.LimitPrice) - 0.01, 2)
            self.Debug(f'Updating limits - Combo 2 {ticket.OrderId}: {newLimit:.2f}')
            fields.LimitPrice = newLimit
            fields.Tag = f'Update #{len(ticket.UpdateRequests) + 1}'
            ticket.Update(fields)

    def ComboLegLimitOrders(self):
        if False:
            i = 10
            return i + 15
        if len(self._openLegLimitOrders) == 0:
            self.Log('Submitting ComboLegLimitOrder')
            for leg in self._orderLegs:
                close = self.Securities[leg.Symbol].Close
                leg.OrderPrice = close * 0.999
            tickets = self.ComboLegLimitOrder(self._orderLegs, quantity=2)
            self._openLegLimitOrders.extend(tickets)
            for leg in self._orderLegs:
                close = self.Securities[leg.Symbol].Close
                leg.OrderPrice = close * 1.001
            tickets = self.ComboLegLimitOrder(self._orderLegs, -2)
            self._openLegLimitOrders.extend(tickets)
        else:
            combo1 = self._openLegLimitOrders[:len(self._orderLegs)]
            combo2 = self._openLegLimitOrders[-len(self._orderLegs):]
            if self.CheckGroupOrdersForFills(combo1, combo2):
                return
            for ticket in combo1:
                newLimit = ticket.Get(OrderField.LimitPrice) + (1 if ticket.Quantity > 0 else -1) * 0.01
                self.Debug(f'Updating limits - Combo #1: {newLimit:.2f}')
                fields = UpdateOrderFields()
                fields.LimitPrice = newLimit
                fields.Tag = f'Update #{len(ticket.UpdateRequests) + 1}'
                ticket.Update(fields)
            for ticket in combo2:
                newLimit = ticket.Get(OrderField.LimitPrice) + (1 if ticket.Quantity > 0 else -1) * 0.01
                self.Debug(f'Updating limits - Combo #2: {newLimit:.2f}')
                fields.LimitPrice = newLimit
                fields.Tag = f'Update #{len(ticket.UpdateRequests) + 1}'
                ticket.Update(fields)

    def OnOrderEvent(self, orderEvent):
        if False:
            print('Hello World!')
        order = self.Transactions.GetOrderById(orderEvent.OrderId)
        if orderEvent.Quantity == 0:
            raise Exception('OrderEvent quantity is Not expected to be 0, it should hold the current order Quantity')
        if orderEvent.Quantity != order.Quantity:
            raise Exception(f'OrderEvent quantity should hold the current order Quantity. Got {orderEvent.Quantity}, expected {order.Quantity}')
        if order.Type == OrderType.ComboLegLimit and orderEvent.LimitPrice == 0:
            raise Exception('OrderEvent.LimitPrice is not expected to be 0 for ComboLegLimitOrder')

    def CheckGroupOrdersForFills(self, combo1, combo2):
        if False:
            return 10
        if all((x.Status == OrderStatus.Filled for x in combo1)):
            self.Log(f'{combo1[0].OrderType}: Canceling combo #2, combo #1 is filled.')
            if any((OrderExtensions.IsOpen(x.Status) for x in combo2)):
                for ticket in combo2:
                    ticket.Cancel('Combo #1 filled.')
            return True
        if all((x.Status == OrderStatus.Filled for x in combo2)):
            self.Log(f'{combo2[0].OrderType}: Canceling combo #1, combo #2 is filled.')
            if any((OrderExtensions.IsOpen(x.Status) for x in combo1)):
                for ticket in combo1:
                    ticket.Cancel('Combo #2 filled.')
            return True
        return False

    def OnEndOfAlgorithm(self):
        if False:
            i = 10
            return i + 15
        filledOrders = self.Transactions.GetOrders(lambda x: x.Status == OrderStatus.Filled).ToList()
        orderTickets = self.Transactions.GetOrderTickets().ToList()
        openOrders = self.Transactions.GetOpenOrders()
        openOrderTickets = self.Transactions.GetOpenOrderTickets().ToList()
        remainingOpenOrders = self.Transactions.GetOpenOrdersRemainingQuantity()
        expectedOrdersCount = 18
        expectedFillsCount = 15
        if len(filledOrders) != expectedFillsCount or len(orderTickets) != expectedOrdersCount:
            raise Exception(f'There were expected {expectedFillsCount} filled orders and {expectedOrdersCount} order tickets, but there were {len(filledOrders)} filled orders and {len(orderTickets)} order tickets')
        filledComboMarketOrders = [x for x in filledOrders if x.Type == OrderType.ComboMarket]
        filledComboLimitOrders = [x for x in filledOrders if x.Type == OrderType.ComboLimit]
        filledComboLegLimitOrders = [x for x in filledOrders if x.Type == OrderType.ComboLegLimit]
        if len(filledComboMarketOrders) != 6 or len(filledComboLimitOrders) != 3 or len(filledComboLegLimitOrders) != 6:
            raise Exception(f'There were expected 6 filled market orders, 3 filled combo limit orders and 6 filled combo leg limit orders, but there were {len(filledComboMarketOrders)} filled market orders, {len(filledComboLimitOrders)} filled combo limit orders and {len(filledComboLegLimitOrders)} filled combo leg limit orders')
        if len(openOrders) != 0 or len(openOrderTickets) != 0:
            raise Exception('No open orders or tickets were expected')
        if remainingOpenOrders != 0:
            raise Exception('No remaining quantity to be filled from open orders was expected')