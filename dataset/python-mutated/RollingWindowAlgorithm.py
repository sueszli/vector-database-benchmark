from AlgorithmImports import *

class RollingWindowAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 1)
        self.SetEndDate(2013, 11, 1)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Daily)
        self.window = RollingWindow[TradeBar](2)
        self.sma = self.SMA('SPY', 5)
        self.sma.Updated += self.SmaUpdated
        self.smaWin = RollingWindow[IndicatorDataPoint](5)

    def SmaUpdated(self, sender, updated):
        if False:
            while True:
                i = 10
        'Adds updated values to rolling window'
        self.smaWin.Add(updated)

    def OnData(self, data):
        if False:
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        self.window.Add(data['SPY'])
        if not (self.window.IsReady and self.smaWin.IsReady):
            return
        currBar = self.window[0]
        pastBar = self.window[1]
        self.Log('Price: {0} -> {1} ... {2} -> {3}'.format(pastBar.Time, pastBar.Close, currBar.Time, currBar.Close))
        currSma = self.smaWin[0]
        pastSma = self.smaWin[self.smaWin.Count - 1]
        self.Log('SMA:   {0} -> {1} ... {2} -> {3}'.format(pastSma.Time, pastSma.Value, currSma.Time, currSma.Value))
        if not self.Portfolio.Invested and currSma.Value > pastSma.Value:
            self.SetHoldings('SPY', 1)