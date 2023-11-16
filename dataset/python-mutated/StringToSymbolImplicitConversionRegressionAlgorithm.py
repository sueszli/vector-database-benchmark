from AlgorithmImports import *

class StringToSymbolImplicitConversionRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 8)
        self.AddEquity('SPY', Resolution.Minute)

    def OnData(self, data):
        if False:
            print('Hello World!')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        try:
            self.MarketOrder('PEPE', 1)
        except Exception as exception:
            if 'This asset symbol (PEPE 0) was not found in your security list' in str(exception) and (not self.Portfolio.Invested):
                self.SetHoldings('SPY', 1)