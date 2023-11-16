from AlgorithmImports import *

class BasicTemplateDailyAlgorithm(QCAlgorithm):
    """Basic template algorithm simply initializes the date range and cash"""

    def Initialize(self):
        if False:
            while True:
                i = 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 17)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Daily)

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1)
            self.Debug('Purchased Stock')