from AlgorithmImports import *

class UniverseOnlyRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2020, 12, 1)
        self.SetEndDate(2020, 12, 12)
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.Universe.ETF('GDVD', self.UniverseSettings, self.FilterUniverse))
        self.selection_done = False

    def FilterUniverse(self, constituents: List[ETFConstituentData]) -> List[Symbol]:
        if False:
            i = 10
            return i + 15
        self.selection_done = True
        return [x.Symbol for x in constituents]

    def OnEndOfAlgorithm(self):
        if False:
            return 10
        if not self.selection_done:
            raise Exception('Universe selection was not performed')