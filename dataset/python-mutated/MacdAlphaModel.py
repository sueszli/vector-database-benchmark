from AlgorithmImports import *

class MacdAlphaModel(AlphaModel):
    """Defines a custom alpha model that uses MACD crossovers. The MACD signal line
    is used to generate up/down insights if it's stronger than the bounce threshold.
    If the MACD signal is within the bounce threshold then a flat price insight is returned."""

    def __init__(self, fastPeriod=12, slowPeriod=26, signalPeriod=9, movingAverageType=MovingAverageType.Exponential, resolution=Resolution.Daily):
        if False:
            return 10
        ' Initializes a new instance of the MacdAlphaModel class\n        Args:\n            fastPeriod: The MACD fast period\n            slowPeriod: The MACD slow period</param>\n            signalPeriod: The smoothing period for the MACD signal\n            movingAverageType: The type of moving average to use in the MACD'
        self.fastPeriod = fastPeriod
        self.slowPeriod = slowPeriod
        self.signalPeriod = signalPeriod
        self.movingAverageType = movingAverageType
        self.resolution = resolution
        self.insightPeriod = Time.Multiply(Extensions.ToTimeSpan(resolution), fastPeriod)
        self.bounceThresholdPercent = 0.01
        self.insightCollection = InsightCollection()
        self.symbolData = {}
        resolutionString = Extensions.GetEnumString(resolution, Resolution)
        movingAverageTypeString = Extensions.GetEnumString(movingAverageType, MovingAverageType)
        self.Name = '{}({},{},{},{},{})'.format(self.__class__.__name__, fastPeriod, slowPeriod, signalPeriod, movingAverageTypeString, resolutionString)

    def Update(self, algorithm, data):
        if False:
            print('Hello World!')
        " Determines an insight for each security based on it's current MACD signal\n        Args:\n            algorithm: The algorithm instance\n            data: The new data available\n        Returns:\n            The new insights generated"
        insights = []
        for (key, sd) in self.symbolData.items():
            if sd.Security.Price == 0:
                continue
            direction = InsightDirection.Flat
            normalized_signal = sd.MACD.Signal.Current.Value / sd.Security.Price
            if normalized_signal > self.bounceThresholdPercent:
                direction = InsightDirection.Up
            elif normalized_signal < -self.bounceThresholdPercent:
                direction = InsightDirection.Down
            if direction == sd.PreviousDirection:
                continue
            sd.PreviousDirection = direction
            if direction == InsightDirection.Flat:
                self.CancelInsights(algorithm, sd.Security.Symbol)
                continue
            insight = Insight.Price(sd.Security.Symbol, self.insightPeriod, direction)
            insights.append(insight)
            self.insightCollection.Add(insight)
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            for i in range(10):
                print('nop')
        'Event fired each time the we add/remove securities from the data feed.\n        This initializes the MACD for each added security and cleans up the indicator for each removed security.\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        for added in changes.AddedSecurities:
            self.symbolData[added.Symbol] = SymbolData(algorithm, added, self.fastPeriod, self.slowPeriod, self.signalPeriod, self.movingAverageType, self.resolution)
        for removed in changes.RemovedSecurities:
            symbol = removed.Symbol
            data = self.symbolData.pop(symbol, None)
            if data is not None:
                algorithm.SubscriptionManager.RemoveConsolidator(symbol, data.Consolidator)
            self.CancelInsights(algorithm, symbol)

    def CancelInsights(self, algorithm, symbol):
        if False:
            while True:
                i = 10
        if not self.insightCollection.ContainsKey(symbol):
            return
        insights = self.insightCollection[symbol]
        algorithm.Insights.Cancel(insights)
        self.insightCollection.Clear([symbol])

class SymbolData:

    def __init__(self, algorithm, security, fastPeriod, slowPeriod, signalPeriod, movingAverageType, resolution):
        if False:
            for i in range(10):
                print('nop')
        self.Security = security
        self.MACD = MovingAverageConvergenceDivergence(fastPeriod, slowPeriod, signalPeriod, movingAverageType)
        self.Consolidator = algorithm.ResolveConsolidator(security.Symbol, resolution)
        algorithm.RegisterIndicator(security.Symbol, self.MACD, self.Consolidator)
        algorithm.WarmUpIndicator(security.Symbol, self.MACD, resolution)
        self.PreviousDirection = None