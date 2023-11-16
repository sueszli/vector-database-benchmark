from AlgorithmImports import *
from Selection.FundamentalUniverseSelectionModel import FundamentalUniverseSelectionModel

class EmaCrossUniverseSelectionModel(FundamentalUniverseSelectionModel):
    """Provides an implementation of FundamentalUniverseSelectionModel that subscribes to
    symbols with the larger delta by percentage between the two exponential moving average"""

    def __init__(self, fastPeriod=100, slowPeriod=300, universeCount=500, universeSettings=None):
        if False:
            return 10
        'Initializes a new instance of the EmaCrossUniverseSelectionModel class\n        Args:\n            fastPeriod: Fast EMA period\n            slowPeriod: Slow EMA period\n            universeCount: Maximum number of members of this universe selection\n            universeSettings: The settings used when adding symbols to the algorithm, specify null to use algorithm.UniverseSettings'
        super().__init__(False, universeSettings)
        self.fastPeriod = fastPeriod
        self.slowPeriod = slowPeriod
        self.universeCount = universeCount
        self.tolerance = 0.01
        self.averages = {}

    def SelectCoarse(self, algorithm, coarse):
        if False:
            i = 10
            return i + 15
        'Defines the coarse fundamental selection function.\n        Args:\n            algorithm: The algorithm instance\n            coarse: The coarse fundamental data used to perform filtering</param>\n        Returns:\n            An enumerable of symbols passing the filter'
        filtered = []
        for cf in coarse:
            if cf.Symbol not in self.averages:
                self.averages[cf.Symbol] = self.SelectionData(cf.Symbol, self.fastPeriod, self.slowPeriod)
            avg = self.averages.get(cf.Symbol)
            if avg.Update(cf.EndTime, cf.AdjustedPrice) and avg.Fast > avg.Slow * (1 + self.tolerance):
                filtered.append(avg)
        filtered = sorted(filtered, key=lambda avg: avg.ScaledDelta, reverse=True)
        return [x.Symbol for x in filtered[:self.universeCount]]

    class SelectionData:

        def __init__(self, symbol, fastPeriod, slowPeriod):
            if False:
                return 10
            self.Symbol = symbol
            self.FastEma = ExponentialMovingAverage(fastPeriod)
            self.SlowEma = ExponentialMovingAverage(slowPeriod)

        @property
        def Fast(self):
            if False:
                for i in range(10):
                    print('nop')
            return float(self.FastEma.Current.Value)

        @property
        def Slow(self):
            if False:
                i = 10
                return i + 15
            return float(self.SlowEma.Current.Value)

        @property
        def ScaledDelta(self):
            if False:
                i = 10
                return i + 15
            return (self.Fast - self.Slow) / ((self.Fast + self.Slow) / 2)

        def Update(self, time, value):
            if False:
                return 10
            return self.SlowEma.Update(time, value) & self.FastEma.Update(time, value)