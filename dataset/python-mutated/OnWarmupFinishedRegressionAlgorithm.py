from AlgorithmImports import *

class OnWarmupFinishedRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Minute)
        self.SetWarmup(timedelta(days=1))
        self._onWarmupFinished = 0

    def OnWarmupFinished(self):
        if False:
            return 10
        self._onWarmupFinished += 1

    def OnEndOfAlgorithm(self):
        if False:
            while True:
                i = 10
        if self._onWarmupFinished != 1:
            raise Exception(f'Unexpected OnWarmupFinished call count {self._onWarmupFinished}')