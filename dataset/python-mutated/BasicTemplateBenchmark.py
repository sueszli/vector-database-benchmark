from AlgorithmImports import *

class BasicTemplateBenchmark(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2000, 1, 1)
        self.SetEndDate(2022, 1, 1)
        self.SetBenchmark(lambda x: 1)
        self.AddEquity('SPY')

    def OnData(self, data):
        if False:
            return 10
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1)
            self.Debug('Purchased Stock')