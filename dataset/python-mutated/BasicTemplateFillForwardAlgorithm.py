from AlgorithmImports import *

class BasicTemplateFillForwardAlgorithm(QCAlgorithm):
    """Basic template algorithm simply initializes the date range and cash"""

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 11, 30)
        self.SetCash(100000)
        self.AddSecurity(SecurityType.Equity, 'ASUR', Resolution.Second)

    def OnData(self, data):
        if False:
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n        \n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            self.SetHoldings('ASUR', 1)