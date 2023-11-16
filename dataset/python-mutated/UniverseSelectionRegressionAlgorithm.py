from AlgorithmImports import *

class UniverseSelectionRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2014, 3, 22)
        self.SetEndDate(2014, 4, 7)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Daily)
        self.AddEquity('GOOG', Resolution.Daily)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        self.delistedSymbols = []
        self.changes = None

    def CoarseSelectionFunction(self, coarse):
        if False:
            return 10
        return [c.Symbol for c in coarse if c.Symbol.Value == 'GOOG' or c.Symbol.Value == 'GOOCV' or c.Symbol.Value == 'GOOAV' or (c.Symbol.Value == 'GOOGL')]

    def OnData(self, data):
        if False:
            return 10
        if self.Transactions.OrdersCount == 0:
            self.MarketOrder('SPY', 100)
        for kvp in data.Delistings:
            self.delistedSymbols.append(kvp.Key)
        if self.changes is None:
            return
        if not all((data.Bars.ContainsKey(x.Symbol) for x in self.changes.AddedSecurities)):
            return
        for security in self.changes.AddedSecurities:
            self.Log('{0}: Added Security: {1}'.format(self.Time, security.Symbol))
            self.MarketOnOpenOrder(security.Symbol, 100)
        for security in self.changes.RemovedSecurities:
            self.Log('{0}: Removed Security: {1}'.format(self.Time, security.Symbol))
            if security.Symbol not in self.delistedSymbols:
                self.Log('Not in delisted: {0}:'.format(security.Symbol))
                self.MarketOnOpenOrder(security.Symbol, -100)
        self.changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            return 10
        self.changes = changes

    def OnOrderEvent(self, orderEvent):
        if False:
            for i in range(10):
                print('nop')
        if orderEvent.Status == OrderStatus.Submitted:
            self.Log('{0}: Submitted: {1}'.format(self.Time, self.Transactions.GetOrderById(orderEvent.OrderId)))
        if orderEvent.Status == OrderStatus.Filled:
            self.Log('{0}: Filled: {1}'.format(self.Time, self.Transactions.GetOrderById(orderEvent.OrderId)))