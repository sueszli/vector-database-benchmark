from AlgorithmImports import *

class CustomShortableProviderRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetCash(1000000)
        self.SetStartDate(2013, 10, 4)
        self.SetEndDate(2013, 10, 6)
        self.spy = self.AddSecurity(SecurityType.Equity, 'SPY', Resolution.Daily)
        self.spy.SetShortableProvider(CustomShortableProvider())

    def OnData(self, data):
        if False:
            return 10
        spyShortableQuantity = self.spy.ShortableProvider.ShortableQuantity(self.spy.Symbol, self.Time)
        if spyShortableQuantity > 1000:
            self.orderId = self.Sell('SPY', int(spyShortableQuantity))

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        transactions = self.Transactions.OrdersCount
        if transactions != 1:
            raise Exception('Algorithm should have just 1 order, but was ' + str(transactions))
        orderQuantity = self.Transactions.GetOrderById(self.orderId).Quantity
        if orderQuantity != -1001:
            raise Exception('Quantity of order ' + str(_orderId) + ' should be ' + str(-1001) + ', but was {orderQuantity}')

class CustomShortableProvider(NullShortableProvider):

    def ShortableQuantity(self, symbol: Symbol, localTime: DateTime):
        if False:
            i = 10
            return i + 15
        if localTime < datetime(2013, 10, 5):
            return 10
        else:
            return 1001