from AlgorithmImports import *
from System.Collections.Generic import List

class EmaCrossUniverseSelectionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2015, 1, 1)
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.Leverage = 2
        self.coarse_count = 10
        self.averages = {}
        self.AddUniverse(self.CoarseSelectionFunction)

    def CoarseSelectionFunction(self, coarse):
        if False:
            print('Hello World!')
        for cf in coarse:
            if cf.Symbol not in self.averages:
                self.averages[cf.Symbol] = SymbolData(cf.Symbol)
            avg = self.averages[cf.Symbol]
            avg.update(cf.EndTime, cf.AdjustedPrice)
        values = list(filter(lambda x: x.is_uptrend, self.averages.values()))
        values.sort(key=lambda x: x.scale, reverse=True)
        for x in values[:self.coarse_count]:
            self.Log('symbol: ' + str(x.symbol.Value) + '  scale: ' + str(x.scale))
        return [x.symbol for x in values[:self.coarse_count]]

    def OnSecuritiesChanged(self, changes):
        if False:
            print('Hello World!')
        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
        for security in changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 0.1)

class SymbolData(object):

    def __init__(self, symbol):
        if False:
            return 10
        self.symbol = symbol
        self.tolerance = 1.01
        self.fast = ExponentialMovingAverage(100)
        self.slow = ExponentialMovingAverage(300)
        self.is_uptrend = False
        self.scale = 0

    def update(self, time, value):
        if False:
            for i in range(10):
                print('nop')
        if self.fast.Update(time, value) and self.slow.Update(time, value):
            fast = self.fast.Current.Value
            slow = self.slow.Current.Value
            self.is_uptrend = fast > slow * self.tolerance
        if self.is_uptrend:
            self.scale = (fast - slow) / ((fast + slow) / 2.0)