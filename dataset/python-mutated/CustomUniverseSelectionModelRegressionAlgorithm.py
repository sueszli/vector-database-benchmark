from AlgorithmImports import *
from Selection.FundamentalUniverseSelectionModel import FundamentalUniverseSelectionModel

class CustomUniverseSelectionModelRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2014, 3, 24)
        self.SetEndDate(2014, 4, 7)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetUniverseSelection(CustomUniverseSelectionModel())

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            for kvp in self.ActiveSecurities:
                self.SetHoldings(kvp.Key, 0.1)

class CustomUniverseSelectionModel(FundamentalUniverseSelectionModel):

    def __init__(self, filterFineData=True, universeSettings=None):
        if False:
            i = 10
            return i + 15
        super().__init__(filterFineData, universeSettings)
        self._selected = False

    def SelectCoarse(self, algorithm, coarse):
        if False:
            while True:
                i = 10
        return [Symbol.Create('AAPL', SecurityType.Equity, Market.USA)]

    def SelectFine(self, algorithm, fine):
        if False:
            while True:
                i = 10
        if not self._selected:
            self._selected = True
            return [x.Symbol for x in fine]
        return Universe.Unchanged