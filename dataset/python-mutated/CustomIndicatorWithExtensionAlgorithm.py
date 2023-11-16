from AlgorithmImports import *
from collections import deque
from math import isclose

class CustomIndicatorWithExtensionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 10, 9)
        self.SetEndDate(2013, 10, 9)
        self.spy = self.AddEquity('SPY', Resolution.Minute).Symbol
        self.sma_values = []
        self.period = 10
        self.sma = self.SMA(self.spy, self.period, Resolution.Minute)
        self.sma.Updated += self.OnSMAUpdated
        self.custom_sma = CustomSimpleMovingAverage('My SMA', self.period)
        self.ext = IndicatorExtensions.Of(self.custom_sma, self.sma)
        self.ext.Updated += self.OnIndicatorExtensionUpdated
        self.sma_minus_custom = IndicatorExtensions.Minus(self.sma, self.custom_sma)
        self.sma_minus_custom.Updated += self.OnMinusUpdated
        self.sma_was_updated = False
        self.custom_sma_was_updated = False
        self.sma_minus_custom_was_updated = False

    def OnSMAUpdated(self, sender, updated):
        if False:
            for i in range(10):
                print('nop')
        self.sma_was_updated = True
        if self.sma.IsReady:
            self.sma_values.append(self.sma.Current.Value)

    def OnIndicatorExtensionUpdated(self, sender, updated):
        if False:
            while True:
                i = 10
        self.custom_sma_was_updated = True
        sma_last_values = self.sma_values[-self.period:]
        expected = sum(sma_last_values) / len(sma_last_values)
        if not isclose(expected, self.custom_sma.Value):
            raise Exception(f'Expected the custom SMA to calculate the moving average of the last {self.period} values of the SMA. Current expected: {expected}. Actual {self.custom_sma.Value}.')
        self.Debug(f'{self.sma.Current.Value} :: {self.custom_sma.Value} :: {updated}')

    def OnMinusUpdated(self, sender, updated):
        if False:
            while True:
                i = 10
        self.sma_minus_custom_was_updated = True
        expected = self.sma.Current.Value - self.custom_sma.Value
        if not isclose(expected, self.sma_minus_custom.Current.Value):
            raise Exception(f'Expected the composite minus indicator to calculate the difference between the SMA and custom SMA indicators. Expected: {expected}. Actual {self.sma_minus_custom.Current.Value}.')

    def OnEndOfAlgorithm(self):
        if False:
            return 10
        if not (self.sma_was_updated and self.custom_sma_was_updated and self.sma_minus_custom_was_updated):
            raise Exception('Expected all indicators to have been updated.')

class CustomSimpleMovingAverage(PythonIndicator):

    def __init__(self, name, period):
        if False:
            for i in range(10):
                print('nop')
        self.Name = name
        self.Value = 0
        self.WarmUpPeriod = period
        self.queue = deque(maxlen=period)

    def Update(self, input: BaseData) -> bool:
        if False:
            while True:
                i = 10
        self.queue.appendleft(input.Value)
        count = len(self.queue)
        self.Value = sum(self.queue) / count
        return count == self.queue.maxlen