from AlgorithmImports import *

class EmptySingleSecuritySecondEquityBenchmark(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2008, 1, 1)
        self.SetEndDate(2008, 6, 1)
        self.SetBenchmark(lambda x: 1)
        self.AddEquity('SPY', Resolution.Second)

    def OnData(self, data):
        if False:
            while True:
                i = 10
        pass