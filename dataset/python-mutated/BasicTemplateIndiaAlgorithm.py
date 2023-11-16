from AlgorithmImports import *

class BasicTemplateIndiaAlgorithm(QCAlgorithm):
    """Basic template framework algorithm uses framework components to define the algorithm."""

    def Initialize(self):
        if False:
            while True:
                i = 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetAccountCurrency('INR')
        self.SetStartDate(2019, 1, 23)
        self.SetEndDate(2019, 10, 31)
        self.SetCash(100000)
        self.AddEquity('YESBANK', Resolution.Minute, Market.India)
        self.Debug('numpy test >>> print numpy.pi: ' + str(np.pi))
        self.DefaultOrderProperties = IndiaOrderProperties(Exchange.NSE)

    def OnData(self, data):
        if False:
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            self.MarketOrder('YESBANK', 1)

    def OnOrderEvent(self, orderEvent):
        if False:
            while True:
                i = 10
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug('Purchased Stock: {0}'.format(orderEvent.Symbol))