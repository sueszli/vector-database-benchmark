from AlgorithmImports import *
from collections import deque

class CustomWarmUpPeriodIndicatorAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.AddEquity('SPY', Resolution.Second)
        self.customNotWarmUp = CSMANotWarmUp('customNotWarmUp', 60)
        self.customWarmUp = CSMAWithWarmUp('customWarmUp', 60)
        self.customNotInherit = CustomSMA('customNotInherit', 60)
        self.csharpIndicator = SimpleMovingAverage('csharpIndicator', 60)
        self.RegisterIndicator('SPY', self.customWarmUp, Resolution.Minute)
        self.RegisterIndicator('SPY', self.customNotWarmUp, Resolution.Minute)
        self.RegisterIndicator('SPY', self.customNotInherit, Resolution.Minute)
        self.RegisterIndicator('SPY', self.csharpIndicator, Resolution.Minute)
        self.WarmUpIndicator('SPY', self.customWarmUp, Resolution.Minute)
        assert self.customWarmUp.IsReady, 'customWarmUp indicator was expected to be ready'
        assert self.customWarmUp.Samples == 60, 'customWarmUp indicator was expected to have processed 60 datapoints already'
        self.WarmUpIndicator('SPY', self.customNotWarmUp, Resolution.Minute)
        assert not self.customNotWarmUp.IsReady, "customNotWarmUp indicator wasn't expected to be warmed up"
        assert self.customNotWarmUp.WarmUpPeriod == 0, 'customNotWarmUp indicator WarmUpPeriod parameter was expected to be 0'
        self.WarmUpIndicator('SPY', self.customNotInherit, Resolution.Minute)
        assert self.customNotInherit.IsReady, 'customNotInherit indicator was expected to be ready'
        assert self.customNotInherit.Samples == 60, 'customNotInherit indicator was expected to have processed 60 datapoints already'
        self.WarmUpIndicator('SPY', self.csharpIndicator, Resolution.Minute)
        assert self.csharpIndicator.IsReady, 'csharpIndicator indicator was expected to be ready'
        assert self.csharpIndicator.Samples == 60, 'csharpIndicator indicator was expected to have processed 60 datapoints already'

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1)
        if self.Time.second == 0:
            diff = abs(self.customNotWarmUp.Current.Value - self.customWarmUp.Current.Value)
            diff += abs(self.customNotInherit.Value - self.customNotWarmUp.Current.Value)
            diff += abs(self.customNotInherit.Value - self.customWarmUp.Current.Value)
            diff += abs(self.csharpIndicator.Current.Value - self.customWarmUp.Current.Value)
            diff += abs(self.csharpIndicator.Current.Value - self.customNotWarmUp.Current.Value)
            diff += abs(self.csharpIndicator.Current.Value - self.customNotInherit.Value)
            assert self.customNotWarmUp.IsReady == (self.customNotWarmUp.Samples >= 60), 'customNotWarmUp indicator was expected to be ready when the number of samples were bigger that its WarmUpPeriod parameter'
            assert diff <= 1e-10 or not self.customNotWarmUp.IsReady, f'The values of the indicators are not the same. Indicators difference is {diff}'

class CSMANotWarmUp(PythonIndicator):

    def __init__(self, name, period):
        if False:
            while True:
                i = 10
        super().__init__()
        self.Name = name
        self.Value = 0
        self.queue = deque(maxlen=period)

    def Update(self, input):
        if False:
            print('Hello World!')
        self.queue.appendleft(input.Value)
        count = len(self.queue)
        self.Value = np.sum(self.queue) / count
        return count == self.queue.maxlen

class CSMAWithWarmUp(CSMANotWarmUp):

    def __init__(self, name, period):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name, period)
        self.WarmUpPeriod = period

class CustomSMA:

    def __init__(self, name, period):
        if False:
            for i in range(10):
                print('nop')
        self.Name = name
        self.Value = 0
        self.queue = deque(maxlen=period)
        self.WarmUpPeriod = period
        self.IsReady = False
        self.Samples = 0

    def Update(self, input):
        if False:
            return 10
        self.Samples += 1
        self.queue.appendleft(input.Value)
        count = len(self.queue)
        self.Value = np.sum(self.queue) / count
        if count == self.queue.maxlen:
            self.IsReady = True
        return self.IsReady