from AlgorithmImports import *
AddReference('System.Collections')
from System.Collections.Generic import List

class UserDefinedUniverseAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetCash(100000)
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2015, 12, 1)
        self.symbols = ['SPY', 'GOOG', 'IBM', 'AAPL', 'MSFT', 'CSCO', 'ADBE', 'WMT']
        self.UniverseSettings.Resolution = Resolution.Hour
        self.AddUniverse('my_universe_name', Resolution.Hour, self.selection)

    def selection(self, time):
        if False:
            for i in range(10):
                print('nop')
        index = time.hour % len(self.symbols)
        return self.symbols[index]

    def OnData(self, slice):
        if False:
            print('Hello World!')
        pass

    def OnSecuritiesChanged(self, changes):
        if False:
            return 10
        for removed in changes.RemovedSecurities:
            if removed.Invested:
                self.Liquidate(removed.Symbol)
        for added in changes.AddedSecurities:
            self.SetHoldings(added.Symbol, 1 / float(len(changes.AddedSecurities)))