from AlgorithmImports import *

class ConstantAlphaModel(AlphaModel):
    """ Provides an implementation of IAlphaModel that always returns the same insight for each security"""

    def __init__(self, type, direction, period, magnitude=None, confidence=None, weight=None):
        if False:
            print('Hello World!')
        'Initializes a new instance of the ConstantAlphaModel class\n        Args:\n            type: The type of insight\n            direction: The direction of the insight\n            period: The period over which the insight with come to fruition\n            magnitude: The predicted change in magnitude as a +- percentage\n            confidence: The confidence in the insight\n            weight: The portfolio weight of the insights'
        self.type = type
        self.direction = direction
        self.period = period
        self.magnitude = magnitude
        self.confidence = confidence
        self.weight = weight
        self.securities = []
        self.insightsTimeBySymbol = {}
        typeString = Extensions.GetEnumString(type, InsightType)
        directionString = Extensions.GetEnumString(direction, InsightDirection)
        self.Name = '{}({},{},{}'.format(self.__class__.__name__, typeString, directionString, strfdelta(period))
        if magnitude is not None:
            self.Name += ',{}'.format(magnitude)
        if confidence is not None:
            self.Name += ',{}'.format(confidence)
        self.Name += ')'

    def Update(self, algorithm, data):
        if False:
            print('Hello World!')
        ' Creates a constant insight for each security as specified via the constructor\n        Args:\n            algorithm: The algorithm instance\n            data: The new data available\n        Returns:\n            The new insights generated'
        insights = []
        for security in self.securities:
            if security.Price != 0 and self.ShouldEmitInsight(algorithm.UtcTime, security.Symbol):
                insights.append(Insight(security.Symbol, self.period, self.type, self.direction, self.magnitude, self.confidence, weight=self.weight))
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            print('Hello World!')
        ' Event fired each time the we add/remove securities from the data feed\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        for added in changes.AddedSecurities:
            self.securities.append(added)
        for removed in changes.RemovedSecurities:
            if removed in self.securities:
                self.securities.remove(removed)
            if removed.Symbol in self.insightsTimeBySymbol:
                self.insightsTimeBySymbol.pop(removed.Symbol)

    def ShouldEmitInsight(self, utcTime, symbol):
        if False:
            print('Hello World!')
        if symbol.IsCanonical():
            return False
        generatedTimeUtc = self.insightsTimeBySymbol.get(symbol)
        if generatedTimeUtc is not None:
            if utcTime - generatedTimeUtc < self.period:
                return False
        self.insightsTimeBySymbol[symbol] = utcTime
        return True

def strfdelta(tdelta):
    if False:
        for i in range(10):
            print('nop')
    d = tdelta.days
    (h, rem) = divmod(tdelta.seconds, 3600)
    (m, s) = divmod(rem, 60)
    return '{}.{:02d}:{:02d}:{:02d}'.format(d, h, m, s)