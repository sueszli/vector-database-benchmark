from AlgorithmImports import *

class InceptionDateSelectionRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 1)
        self.SetEndDate(2013, 10, 31)
        self.SetCash(100000)
        self.changes = None
        self.UniverseSettings.Resolution = Resolution.Hour
        self.AddUniverseSelection(CustomUniverseSelectionModel('my-custom-universe', lambda dt: ['IBM'] if dt.day % 7 == 0 else []))
        self.AddUniverseSelection(InceptionDateUniverseSelectionModel('spy-inception', {'SPY': self.StartDate + timedelta(5)}))

    def OnData(self, slice):
        if False:
            i = 10
            return i + 15
        if self.changes is None:
            return
        for security in self.changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 0.5)
        self.changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            return 10
        for security in changes.RemovedSecurities:
            self.Liquidate(security.Symbol, 'Removed from Universe')
        self.changes = changes