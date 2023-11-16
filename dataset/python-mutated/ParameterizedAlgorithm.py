from AlgorithmImports import *

class ParameterizedAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity('SPY')
        fast_period = self.GetParameter('ema-fast', 100)
        slow_period = self.GetParameter('ema-slow', 200)
        self.fast = self.EMA('SPY', fast_period)
        self.slow = self.EMA('SPY', slow_period)

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if not self.fast.IsReady or not self.slow.IsReady:
            return
        fast = self.fast.Current.Value
        slow = self.slow.Current.Value
        if fast > slow * 1.001:
            self.SetHoldings('SPY', 1)
        elif fast < slow * 0.999:
            self.Liquidate('SPY')