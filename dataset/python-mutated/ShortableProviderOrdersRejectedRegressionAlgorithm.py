from AlgorithmImports import *

class RegressionTestShortableProvider(LocalDiskShortableProvider):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(SecurityType.Equity, 'testbrokerage', Market.USA)

class ShortableProviderOrdersRejectedRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.ordersAllowed = []
        self.ordersDenied = []
        self.initialize = False
        self.invalidatedAllowedOrder = False
        self.invalidatedNewOrderWithPortfolioHoldings = False
        self.SetStartDate(2013, 10, 4)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(10000000)
        self.spy = self.AddEquity('SPY', Resolution.Minute)
        self.aig = self.AddEquity('AIG', Resolution.Minute)
        self.spy.SetShortableProvider(RegressionTestShortableProvider())
        self.aig.SetShortableProvider(RegressionTestShortableProvider())

    def OnData(self, data):
        if False:
            print('Hello World!')
        if not self.initialize:
            self.HandleOrder(self.LimitOrder(self.spy.Symbol, -1001, 10000))
            self.HandleOrder(self.LimitOrder(self.spy.Symbol, -1000, 10000))
            self.HandleOrder(self.LimitOrder(self.spy.Symbol, -10, 0.01))
            self.initialize = True
            return
        if not self.invalidatedAllowedOrder:
            if len(self.ordersAllowed) != 1:
                raise Exception(f'Expected 1 successful order, found: {len(self.ordersAllowed)}')
            if len(self.ordersDenied) != 2:
                raise Exception(f'Expected 2 failed orders, found: {len(self.ordersDenied)}')
            allowedOrder = self.ordersAllowed[0]
            orderUpdate = UpdateOrderFields()
            orderUpdate.LimitPrice = 0.01
            orderUpdate.Quantity = -1001
            orderUpdate.Tag = 'Testing updating and exceeding maximum quantity'
            response = allowedOrder.Update(orderUpdate)
            if response.ErrorCode != OrderResponseErrorCode.ExceedsShortableQuantity:
                raise Exception(f'Expected order to fail due to exceeded shortable quantity, found: {response.ErrorCode}')
            cancelResponse = allowedOrder.Cancel()
            if cancelResponse.IsError:
                raise Exception('Expected to be able to cancel open order after bad qty update')
            self.invalidatedAllowedOrder = True
            self.ordersDenied.clear()
            self.ordersAllowed.clear()
            return
        if not self.invalidatedNewOrderWithPortfolioHoldings:
            self.HandleOrder(self.MarketOrder(self.spy.Symbol, -1000))
            spyShares = self.Portfolio[self.spy.Symbol].Quantity
            if spyShares != -1000:
                raise Exception(f'Expected -1000 shares in portfolio, found: {spyShares}')
            self.HandleOrder(self.LimitOrder(self.spy.Symbol, -1, 0.01))
            if len(self.ordersDenied) != 1:
                raise Exception(f'Expected limit order to fail due to existing holdings, but found {len(self.ordersDenied)} failures')
            self.ordersAllowed.clear()
            self.ordersDenied.clear()
            self.HandleOrder(self.MarketOrder(self.aig.Symbol, -1001))
            if len(self.ordersAllowed) != 1:
                raise Exception(f'Expected market order of -1001 BAC to not fail')
            self.invalidatedNewOrderWithPortfolioHoldings = True

    def HandleOrder(self, orderTicket):
        if False:
            for i in range(10):
                print('nop')
        if orderTicket.SubmitRequest.Status == OrderRequestStatus.Error:
            self.ordersDenied.append(orderTicket)
            return
        self.ordersAllowed.append(orderTicket)