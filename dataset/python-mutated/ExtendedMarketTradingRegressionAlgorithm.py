from AlgorithmImports import *

class ExtendedMarketTradingRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.spy = self.AddEquity('SPY', Resolution.Minute, Market.USA, True, 1, True)
        self._lastAction = None

    def OnData(self, data):
        if False:
            print('Hello World!')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if self._lastAction is not None and self._lastAction.date() == self.Time.date():
            return
        spyBar = data.Bars['SPY']
        if not self.InMarketHours():
            self.LimitOrder('SPY', 10, spyBar.Low)
            self._lastAction = self.Time

    def OnOrderEvent(self, orderEvent):
        if False:
            print('Hello World!')
        self.Log(str(orderEvent))
        if self.InMarketHours():
            raise Exception('Order processed during market hours.')

    def InMarketHours(self):
        if False:
            for i in range(10):
                print('nop')
        now = self.Time.time()
        open = time(9, 30, 0)
        close = time(16, 0, 0)
        return open < now and close > now