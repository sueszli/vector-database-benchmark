from AlgorithmImports import *

class CustomBrokerageModelRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetBrokerageModel(CustomBrokerageModel())
        self.AddEquity('SPY', Resolution.Daily)
        self.AddEquity('AIG', Resolution.Daily)
        self.updateRequestSubmitted = False
        if self.BrokerageModel.DefaultMarkets[SecurityType.Equity] != Market.USA:
            raise Exception(f'The default market for Equity should be {Market.USA}')
        if self.BrokerageModel.DefaultMarkets[SecurityType.Crypto] != Market.Binance:
            raise Exception(f'The default market for Crypto should be {Market.Binance}')

    def OnData(self, slice):
        if False:
            print('Hello World!')
        if not self.Portfolio.Invested:
            self.MarketOrder('SPY', 100.0)
            self.aigTicket = self.MarketOrder('AIG', 100.0)

    def OnOrderEvent(self, orderEvent):
        if False:
            print('Hello World!')
        spyTicket = self.Transactions.GetOrderTicket(orderEvent.OrderId)
        if self.updateRequestSubmitted == False:
            updateOrderFields = UpdateOrderFields()
            updateOrderFields.Quantity = spyTicket.Quantity + 10
            spyTicket.Update(updateOrderFields)
            self.spyTicket = spyTicket
            self.updateRequestSubmitted = True

    def OnEndOfAlgorithm(self):
        if False:
            i = 10
            return i + 15
        submitExpectedMessage = 'BrokerageModel declared unable to submit order: [2] Information - Code:  - Symbol AIG can not be submitted'
        if self.aigTicket.SubmitRequest.Response.ErrorMessage != submitExpectedMessage:
            raise Exception(f'Order with ID: {self.aigTicket.OrderId} should not have submitted symbol AIG')
        updateExpectedMessage = 'OrderID: 1 Information - Code:  - This order can not be updated'
        if self.spyTicket.UpdateRequests[0].Response.ErrorMessage != updateExpectedMessage:
            raise Exception(f'Order with ID: {self.spyTicket.OrderId} should have been updated')

class CustomBrokerageModel(DefaultBrokerageModel):
    DefaultMarkets = {SecurityType.Equity: Market.USA, SecurityType.Crypto: Market.Binance}

    def CanSubmitOrder(self, security: SecurityType, order: Order, message: BrokerageMessageEvent):
        if False:
            print('Hello World!')
        if security.Symbol.Value == 'AIG':
            message = BrokerageMessageEvent(BrokerageMessageType.Information, '', 'Symbol AIG can not be submitted')
            return (False, message)
        return (True, None)

    def CanUpdateOrder(self, security: SecurityType, order: Order, request: UpdateOrderRequest, message: BrokerageMessageEvent):
        if False:
            i = 10
            return i + 15
        message = BrokerageMessageEvent(BrokerageMessageType.Information, '', 'This order can not be updated')
        return (False, message)