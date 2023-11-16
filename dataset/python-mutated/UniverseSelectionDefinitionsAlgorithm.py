from AlgorithmImports import *

class UniverseSelectionDefinitionsAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.UniverseSettings.Resolution = Resolution.Hour
        self.UniverseSettings.MinimumTimeInUniverse = timedelta(minutes=30)
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddUniverse(self.Universe.Top(50))
        self.changes = None

    def OnData(self, data):
        if False:
            while True:
                i = 10
        if self.changes is None:
            return
        for security in self.changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
        for security in self.changes.AddedSecurities:
            if not security.Invested:
                self.MarketOrder(security.Symbol, 10)
        self.changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            for i in range(10):
                print('nop')
        self.changes = changes