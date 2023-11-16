from AlgorithmImports import *

class CustomBenchmarkAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Second)
        self.SetBenchmark(Symbol.Create('AAPL', SecurityType.Equity, Market.USA))

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1)
            self.Debug('Purchased Stock')
        tupleResult = SymbolCache.TryGetSymbol('AAPL', None)
        if tupleResult[0]:
            raise Exception('Benchmark Symbol is not expected to be added to the Symbol cache')