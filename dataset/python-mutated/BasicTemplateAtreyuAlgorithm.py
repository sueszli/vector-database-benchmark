from AlgorithmImports import *

class BasicTemplateAtreyuAlgorithm(QCAlgorithm):
    """Basic template algorithm simply initializes the date range and cash"""

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.Atreyu)
        self.AddEquity('SPY', Resolution.Minute)
        self.DefaultOrderProperties = AtreyuOrderProperties()
        self.DefaultOrderProperties.TimeInForce = TimeInForce.Day

    def OnData(self, data):
        if False:
            return 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 0.25)
            self.Debug('Purchased SPY!')