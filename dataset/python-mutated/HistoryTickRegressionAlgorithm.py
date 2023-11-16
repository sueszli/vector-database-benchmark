from AlgorithmImports import *

class HistoryTickRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 10, 11)
        self.SetEndDate(2013, 10, 11)
        self._symbol = self.AddEquity('SPY', Resolution.Tick).Symbol

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        history = list(self.History[Tick](self._symbol, timedelta(days=1), Resolution.Tick))
        quotes = [x for x in history if x.TickType == TickType.Quote]
        trades = [x for x in history if x.TickType == TickType.Trade]
        if not quotes or not trades:
            raise Exception('Expected to find at least one tick of each type (quote and trade)')