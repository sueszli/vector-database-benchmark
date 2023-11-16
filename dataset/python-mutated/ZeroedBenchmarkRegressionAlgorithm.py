from AlgorithmImports import *

class ZeroedBenchmarkRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetCash(100000)
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 8)
        self.AddEquity('SPY', Resolution.Hour)
        self.SetBrokerageModel(TestBrokerageModel())

    def OnData(self, data):
        if False:
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1)

class TestBrokerageModel(DefaultBrokerageModel):

    def GetBenchmark(self, securities):
        if False:
            print('Hello World!')
        return FuncBenchmark(self.func)

    def func(self, datetime):
        if False:
            while True:
                i = 10
        return 0