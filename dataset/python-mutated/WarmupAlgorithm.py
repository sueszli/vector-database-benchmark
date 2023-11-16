from AlgorithmImports import *

class WarmupAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Second)
        fast_period = 60
        slow_period = 3600
        self.fast = self.EMA('SPY', fast_period)
        self.slow = self.EMA('SPY', slow_period)
        self.SetWarmup(slow_period)
        self.first = True

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if self.first and (not self.IsWarmingUp):
            self.first = False
            self.Log('Fast: {0}'.format(self.fast.Samples))
            self.Log('Slow: {0}'.format(self.slow.Samples))
        if self.fast.Current.Value > self.slow.Current.Value:
            self.SetHoldings('SPY', 1)
        else:
            self.SetHoldings('SPY', -1)