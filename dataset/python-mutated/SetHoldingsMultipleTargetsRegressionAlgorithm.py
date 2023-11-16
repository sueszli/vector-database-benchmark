from AlgorithmImports import *

class SetHoldingsMultipleTargetsRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self._spy = self.AddEquity('SPY', Resolution.Minute, Market.USA, False, 1).Symbol
        self._ibm = self.AddEquity('IBM', Resolution.Minute, Market.USA, False, 1).Symbol

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            self.SetHoldings([PortfolioTarget(self._spy, 0.8), PortfolioTarget(self._ibm, 0.2)])
        else:
            self.SetHoldings([PortfolioTarget(self._ibm, 0.8), PortfolioTarget(self._spy, 0.2)])