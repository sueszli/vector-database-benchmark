from AlgorithmImports import *

class EmaCrossAlphaModel(AlphaModel):
    """Alpha model that uses an EMA cross to create insights"""

    def __init__(self, fastPeriod=12, slowPeriod=26, resolution=Resolution.Daily):
        if False:
            print('Hello World!')
        'Initializes a new instance of the EmaCrossAlphaModel class\n        Args:\n            fastPeriod: The fast EMA period\n            slowPeriod: The slow EMA period'
        self.fastPeriod = fastPeriod
        self.slowPeriod = slowPeriod
        self.resolution = resolution
        self.predictionInterval = Time.Multiply(Extensions.ToTimeSpan(resolution), fastPeriod)
        self.symbolDataBySymbol = {}
        resolutionString = Extensions.GetEnumString(resolution, Resolution)
        self.Name = '{}({},{},{})'.format(self.__class__.__name__, fastPeriod, slowPeriod, resolutionString)

    def Update(self, algorithm, data):
        if False:
            while True:
                i = 10
        'Updates this alpha model with the latest data from the algorithm.\n        This is called each time the algorithm receives data for subscribed securities\n        Args:\n            algorithm: The algorithm instance\n            data: The new data available\n        Returns:\n            The new insights generated'
        insights = []
        for (symbol, symbolData) in self.symbolDataBySymbol.items():
            if symbolData.Fast.IsReady and symbolData.Slow.IsReady:
                if symbolData.FastIsOverSlow:
                    if symbolData.Slow > symbolData.Fast:
                        insights.append(Insight.Price(symbolData.Symbol, self.predictionInterval, InsightDirection.Down))
                elif symbolData.SlowIsOverFast:
                    if symbolData.Fast > symbolData.Slow:
                        insights.append(Insight.Price(symbolData.Symbol, self.predictionInterval, InsightDirection.Up))
            symbolData.FastIsOverSlow = symbolData.Fast > symbolData.Slow
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            print('Hello World!')
        'Event fired each time the we add/remove securities from the data feed\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        for added in changes.AddedSecurities:
            symbolData = self.symbolDataBySymbol.get(added.Symbol)
            if symbolData is None:
                symbolData = SymbolData(added, self.fastPeriod, self.slowPeriod, algorithm, self.resolution)
                self.symbolDataBySymbol[added.Symbol] = symbolData
            else:
                symbolData.Fast.Reset()
                symbolData.Slow.Reset()
        for removed in changes.RemovedSecurities:
            data = self.symbolDataBySymbol.pop(removed.Symbol, None)
            if data is not None:
                data.RemoveConsolidators()

class SymbolData:
    """Contains data specific to a symbol required by this model"""

    def __init__(self, security, fastPeriod, slowPeriod, algorithm, resolution):
        if False:
            for i in range(10):
                print('nop')
        self.Security = security
        self.Symbol = security.Symbol
        self.algorithm = algorithm
        self.FastConsolidator = algorithm.ResolveConsolidator(security.Symbol, resolution)
        self.SlowConsolidator = algorithm.ResolveConsolidator(security.Symbol, resolution)
        algorithm.SubscriptionManager.AddConsolidator(security.Symbol, self.FastConsolidator)
        algorithm.SubscriptionManager.AddConsolidator(security.Symbol, self.SlowConsolidator)
        self.Fast = ExponentialMovingAverage(security.Symbol, fastPeriod, ExponentialMovingAverage.SmoothingFactorDefault(fastPeriod))
        self.Slow = ExponentialMovingAverage(security.Symbol, slowPeriod, ExponentialMovingAverage.SmoothingFactorDefault(slowPeriod))
        algorithm.RegisterIndicator(security.Symbol, self.Fast, self.FastConsolidator)
        algorithm.RegisterIndicator(security.Symbol, self.Slow, self.SlowConsolidator)
        algorithm.WarmUpIndicator(security.Symbol, self.Fast, resolution)
        algorithm.WarmUpIndicator(security.Symbol, self.Slow, resolution)
        self.FastIsOverSlow = False

    def RemoveConsolidators(self):
        if False:
            while True:
                i = 10
        self.algorithm.SubscriptionManager.RemoveConsolidator(self.Security.Symbol, self.FastConsolidator)
        self.algorithm.SubscriptionManager.RemoveConsolidator(self.Security.Symbol, self.SlowConsolidator)

    @property
    def SlowIsOverFast(self):
        if False:
            i = 10
            return i + 15
        return not self.FastIsOverSlow