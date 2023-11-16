from AlgorithmImports import *

class OnWarmupFinishedNoWarmup(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.AddEquity('SPY', Resolution.Minute)
        self._onWarmupFinished = 0

    def OnWarmupFinished(self):
        if False:
            for i in range(10):
                print('nop')
        self._onWarmupFinished += 1

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        if self._onWarmupFinished != 1:
            raise Exception(f'Unexpected OnWarmupFinished call count {self._onWarmupFinished}')