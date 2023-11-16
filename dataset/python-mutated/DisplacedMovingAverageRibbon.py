from AlgorithmImports import *

class DisplacedMovingAverageRibbon(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2015, 1, 1)
        self.spy = self.AddEquity('SPY', Resolution.Daily).Symbol
        count = 6
        offset = 5
        period = 15
        self.ribbon = []
        self.sma = SimpleMovingAverage(period)
        for x in range(count):
            delay = Delay(offset * (x + 1))
            delayedSma = IndicatorExtensions.Of(delay, self.sma)
            self.RegisterIndicator(self.spy, delayedSma, Resolution.Daily)
            self.ribbon.append(delayedSma)
        self.previous = datetime.min
        for i in self.ribbon:
            self.PlotIndicator('Ribbon', i)

    def OnData(self, data):
        if False:
            return 10
        if data[self.spy] is None:
            return
        if not all((x.IsReady for x in self.ribbon)):
            return
        if self.previous.date() == self.Time.date():
            return
        self.Plot('Ribbon', 'Price', data[self.spy].Price)
        values = [x.Current.Value for x in self.ribbon]
        holding = self.Portfolio[self.spy]
        if holding.Quantity <= 0 and self.IsAscending(values):
            self.SetHoldings(self.spy, 1.0)
        elif holding.Quantity > 0 and self.IsDescending(values):
            self.Liquidate(self.spy)
        self.previous = self.Time

    def IsAscending(self, values):
        if False:
            i = 10
            return i + 15
        last = None
        for val in values:
            if last is None:
                last = val
                continue
            if last < val:
                return False
            last = val
        return True

    def IsDescending(self, values):
        if False:
            i = 10
            return i + 15
        last = None
        for val in values:
            if last is None:
                last = val
                continue
            if last > val:
                return False
            last = val
        return True