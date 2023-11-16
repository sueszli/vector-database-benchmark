from AlgorithmImports import *

class AutoRegressiveIntegratedMovingAverageRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 1, 7)
        self.SetEndDate(2013, 12, 11)
        self.EnableAutomaticIndicatorWarmUp = True
        self.AddEquity('SPY', Resolution.Daily)
        self.arima = self.ARIMA('SPY', 1, 1, 1, 50)
        self.ar = self.ARIMA('SPY', 1, 1, 0, 50)

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if self.arima.IsReady:
            if abs(self.arima.Current.Value - self.ar.Current.Value) > 1:
                if self.arima.Current.Value > self.last:
                    self.MarketOrder('SPY', 1)
                else:
                    self.MarketOrder('SPY', -1)
            self.last = self.arima.Current.Value