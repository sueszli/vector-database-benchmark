from AlgorithmImports import *

class WeeklyUniverseSelectionRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetCash(100000)
        self.SetStartDate(2013, 10, 1)
        self.SetEndDate(2013, 10, 31)
        self.UniverseSettings.Resolution = Resolution.Hour
        self.AddUniverse('my-custom-universe', lambda dt: ['IBM'] if dt.day % 7 == 0 else [])

    def OnData(self, slice):
        if False:
            for i in range(10):
                print('nop')
        if self.changes is None:
            return
        for security in self.changes.RemovedSecurities:
            if security.Invested:
                self.Log('{} Liquidate {}'.format(self.Time, security.Symbol))
                self.Liquidate(security.Symbol)
        for security in self.changes.AddedSecurities:
            if not security.Invested:
                self.Log('{} Buy {}'.format(self.Time, security.Symbol))
                self.SetHoldings(security.Symbol, 1)
        self.changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            for i in range(10):
                print('nop')
        self.changes = changes