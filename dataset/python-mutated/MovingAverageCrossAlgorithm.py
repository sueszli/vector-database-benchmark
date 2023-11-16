from AlgorithmImports import *

class MovingAverageCrossAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2015, 1, 1)
        self.SetCash(100000)
        self.AddEquity('SPY')
        self.fast = self.EMA('SPY', 15, Resolution.Daily)
        self.slow = self.EMA('SPY', 30, Resolution.Daily)
        self.previous = None

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if not self.slow.IsReady:
            return
        if self.previous is not None and self.previous.date() == self.Time.date():
            return
        tolerance = 0.00015
        holdings = self.Portfolio['SPY'].Quantity
        if holdings <= 0:
            if self.fast.Current.Value > self.slow.Current.Value * (1 + tolerance):
                self.Log('BUY  >> {0}'.format(self.Securities['SPY'].Price))
                self.SetHoldings('SPY', 1.0)
        if holdings > 0 and self.fast.Current.Value < self.slow.Current.Value:
            self.Log('SELL >> {0}'.format(self.Securities['SPY'].Price))
            self.Liquidate('SPY')
        self.previous = self.Time