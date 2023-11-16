from AlgorithmImports import *
from collections import deque

class SecurityDynamicPropertyPythonClassAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 7)
        self.spy = self.AddEquity('SPY', Resolution.Minute)
        customSMA = CustomSimpleMovingAverage('custom', 60)
        self.spy.CustomSMA = customSMA
        customSMA.Security = self.spy
        self.RegisterIndicator(self.spy.Symbol, self.spy.CustomSMA, Resolution.Minute)

    def OnWarmupFinished(self) -> None:
        if False:
            return 10
        if type(self.spy.CustomSMA) != CustomSimpleMovingAverage:
            raise Exception('spy.CustomSMA is not an instance of CustomSimpleMovingAverage')
        if self.spy.CustomSMA.Security is None:
            raise Exception('spy.CustomSMA.Security is None')
        else:
            self.Debug(f'spy.CustomSMA.Security.Symbol: {self.spy.CustomSMA.Security.Symbol}')

    def OnData(self, slice: Slice) -> None:
        if False:
            while True:
                i = 10
        if self.spy.CustomSMA.IsReady:
            self.Debug(f'CustomSMA: {self.spy.CustomSMA.Current.Value}')

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