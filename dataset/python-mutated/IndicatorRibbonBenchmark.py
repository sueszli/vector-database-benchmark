from AlgorithmImports import *

class IndicatorRibbonBenchmark(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2018, 1, 1)
        self.spy = self.AddEquity('SPY', Resolution.Minute).Symbol
        count = 50
        offset = 5
        period = 15
        self.ribbon = []
        self.sma = SimpleMovingAverage(period)
        for x in range(count):
            delay = Delay(offset * (x + 1))
            delayedSma = IndicatorExtensions.Of(delay, self.sma)
            self.RegisterIndicator(self.spy, delayedSma, Resolution.Daily)
            self.ribbon.append(delayedSma)

    def OnData(self, data):
        if False:
            while True:
                i = 10
        if not all((x.IsReady for x in self.ribbon)):
            return
        for x in self.ribbon:
            value = x.Current.Value