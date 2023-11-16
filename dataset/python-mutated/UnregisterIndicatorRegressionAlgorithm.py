from AlgorithmImports import *

class UnregisterIndicatorRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        spy = self.AddEquity('SPY')
        ibm = self.AddEquity('IBM')
        self._symbols = [spy.Symbol, ibm.Symbol]
        self._trin = self.TRIN(self._symbols, Resolution.Minute)
        self._trin2 = None

    def OnData(self, data):
        if False:
            return 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if self._trin.IsReady:
            self._trin.Reset()
            self.UnregisterIndicator(self._trin)
            self._trin2 = self.TRIN(self._symbols, Resolution.Hour)
        if not self._trin2 is None and self._trin2.IsReady:
            if self._trin.IsReady:
                raise ValueError('Indicator should of stop getting updates!')
            if not self.Portfolio.Invested:
                self.SetHoldings(self._symbols[0], 0.5)
                self.SetHoldings(self._symbols[1], 0.5)