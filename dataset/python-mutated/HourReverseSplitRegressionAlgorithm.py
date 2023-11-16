from AlgorithmImports import *

class HourReverseSplitRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 11, 7)
        self.SetEndDate(2013, 11, 8)
        self.SetCash(100000)
        self.SetBenchmark(lambda x: 0)
        self.symbol = self.AddEquity('VXX.1', Resolution.Hour).Symbol

    def OnData(self, slice):
        if False:
            print('Hello World!')
        if slice.Bars.Count == 0:
            return
        if not self.Portfolio.Invested and self.Time.date() == self.EndDate.date():
            self.Buy(self.symbol, 1)