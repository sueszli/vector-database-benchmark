from AlgorithmImports import *
from collections import deque

class CustomIndicatorAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.AddEquity('SPY', Resolution.Second)
        self.sma = self.SMA('SPY', 60, Resolution.Minute)
        self.custom = CustomSimpleMovingAverage('custom', 60)
        self.custom.Updated += self.CustomUpdated
        self.customWindow = RollingWindow[IndicatorDataPoint](5)
        self.RegisterIndicator('SPY', self.custom, Resolution.Minute)
        self.PlotIndicator('cSMA', self.custom)

    def CustomUpdated(self, sender, updated):
        if False:
            i = 10
            return i + 15
        self.customWindow.Add(updated)

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1)
        if self.Time.second == 0:
            self.Log(f'   sma -> IsReady: {self.sma.IsReady}. Value: {self.sma.Current.Value}')
            self.Log(f'custom -> IsReady: {self.custom.IsReady}. Value: {self.custom.Value}')
        diff = abs(self.custom.Value - self.sma.Current.Value)
        if diff > 1e-10:
            self.Quit(f'Quit: indicators difference is {diff}')

    def OnEndOfAlgorithm(self):
        if False:
            while True:
                i = 10
        for item in self.customWindow:
            self.Log(f'{item}')

class CustomSimpleMovingAverage(PythonIndicator):

    def __init__(self, name, period):
        if False:
            print('Hello World!')
        super().__init__()
        self.Name = name
        self.Value = 0
        self.queue = deque(maxlen=period)

    def Update(self, input):
        if False:
            for i in range(10):
                print('nop')
        self.queue.appendleft(input.Value)
        count = len(self.queue)
        self.Value = np.sum(self.queue) / count
        return count == self.queue.maxlen