from AlgorithmImports import *

class SetHoldingsRegressionAlgorithm(QCAlgorithm):
    """Basic template algorithm simply initializes the date range and cash"""

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
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 0.1)
            self.SetHoldings('SPY', np.float(0.2))
            self.SetHoldings('SPY', np.float64(0.3))
            self.SetHoldings('SPY', 1)