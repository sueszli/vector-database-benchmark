from AlgorithmImports import *

class AddRemoveSecurityRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity('SPY')
        self._lastAction = None

    def OnData(self, data):
        if False:
            return 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if self._lastAction is not None and self._lastAction.date() == self.Time.date():
            return
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 0.5)
            self._lastAction = self.Time
        if self.Time.weekday() == 1:
            self.AddEquity('AIG')
            self.AddEquity('BAC')
            self._lastAction = self.Time
        if self.Time.weekday() == 2:
            self.SetHoldings('AIG', 0.25)
            self.SetHoldings('BAC', 0.25)
            self._lastAction = self.Time
        if self.Time.weekday() == 3:
            self.RemoveSecurity('AIG')
            self.RemoveSecurity('BAC')
            self._lastAction = self.Time

    def OnOrderEvent(self, orderEvent):
        if False:
            while True:
                i = 10
        if orderEvent.Status == OrderStatus.Submitted:
            self.Debug('{0}: Submitted: {1}'.format(self.Time, self.Transactions.GetOrderById(orderEvent.OrderId)))
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug('{0}: Filled: {1}'.format(self.Time, self.Transactions.GetOrderById(orderEvent.OrderId)))