from AlgorithmImports import *

class HourSplitRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2014, 6, 6)
        self.SetEndDate(2014, 6, 9)
        self.SetCash(100000)
        self.SetBenchmark(lambda x: 0)
        self.symbol = self.AddEquity('AAPL', Resolution.Hour).Symbol

    def OnData(self, slice):
        if False:
            return 10
        if slice.Bars.Count == 0:
            return
        if not self.Portfolio.Invested and self.Time.date() == self.EndDate.date():
            self.Buy(self.symbol, 1)