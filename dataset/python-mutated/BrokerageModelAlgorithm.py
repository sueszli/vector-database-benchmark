from AlgorithmImports import *

class BrokerageModelAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetCash(100000)
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.AddEquity('SPY', Resolution.Second)
        self.SetBrokerageModel(MinimumAccountBalanceBrokerageModel(self, 500.0))
        self.last = 1

    def OnData(self, slice):
        if False:
            while True:
                i = 10
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', self.last)
            if self.Portfolio['SPY'].Quantity == 0:
                self.Debug(str(self.Time) + ' - Failed to purchase stock')
                self.last *= 0.95
            else:
                self.Debug('{} - Purchased Stock @ SetHoldings( {} )'.format(self.Time, self.last))

class MinimumAccountBalanceBrokerageModel(DefaultBrokerageModel):
    """Custom brokerage model that requires clients to maintain a minimum cash balance"""

    def __init__(self, algorithm, minimumAccountBalance):
        if False:
            i = 10
            return i + 15
        self.algorithm = algorithm
        self.minimumAccountBalance = minimumAccountBalance

    def CanSubmitOrder(self, security, order, message):
        if False:
            i = 10
            return i + 15
        'Prevent orders which would bring the account below a minimum cash balance'
        message = None
        orderCost = order.GetValue(security)
        cash = self.algorithm.Portfolio.Cash
        cashAfterOrder = cash - orderCost
        if cashAfterOrder < self.minimumAccountBalance:
            message = BrokerageMessageEvent(BrokerageMessageType.Warning, 'InsufficientRemainingCapital', 'Account must maintain a minimum of ${0} USD at all times. Order ID: {1}'.format(self.minimumAccountBalance, order.Id))
            self.algorithm.Error(str(message))
            return False
        return True