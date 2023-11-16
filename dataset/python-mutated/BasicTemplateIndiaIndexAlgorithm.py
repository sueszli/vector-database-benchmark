from AlgorithmImports import *

class BasicTemplateIndiaIndexAlgorithm(QCAlgorithm):
    """Basic template framework algorithm uses framework components to define the algorithm."""

    def Initialize(self):
        if False:
            print('Hello World!')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetAccountCurrency('INR')
        self.SetStartDate(2019, 1, 1)
        self.SetEndDate(2019, 1, 5)
        self.SetCash(1000000)
        self.Nifty = self.AddIndex('NIFTY50', Resolution.Minute, Market.India).Symbol
        self.NiftyETF = self.AddEquity('JUNIORBEES', Resolution.Minute, Market.India).Symbol
        self.DefaultOrderProperties = IndiaOrderProperties(Exchange.NSE)
        self._emaSlow = self.EMA(self.Nifty, 80)
        self._emaFast = self.EMA(self.Nifty, 200)
        self.Debug('numpy test >>> print numpy.pi: ' + str(np.pi))

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not data.Bars.ContainsKey(self.Nifty) or not data.Bars.ContainsKey(self.NiftyETF):
            return
        if not self._emaSlow.IsReady:
            return
        if self._emaFast > self._emaSlow:
            if not self.Portfolio.Invested:
                self.marketTicket = self.MarketOrder(self.NiftyETF, 1)
        else:
            self.Liquidate()

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        if self.Portfolio[self.Nifty].TotalSaleVolume > 0:
            raise Exception('Index is not tradable.')