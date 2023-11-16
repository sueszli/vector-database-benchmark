from AlgorithmImports import *
from Portfolio.EqualWeightingPortfolioConstructionModel import *

class AccumulativeInsightPortfolioConstructionModel(EqualWeightingPortfolioConstructionModel):
    """Provides an implementation of IPortfolioConstructionModel that allocates percent of account
    to each insight, defaulting to 3%.
    For insights of direction InsightDirection.Up, long targets are returned and
    for insights of direction InsightDirection.Down, short targets are returned.
    By default, no rebalancing shall be done.
    Rules:
        1. On active Up insight, increase position size by percent
        2. On active Down insight, decrease position size by percent
        3. On active Flat insight, move by percent towards 0
        4. On expired insight, and no other active insight, emits a 0 target"""

    def __init__(self, rebalance=None, portfolioBias=PortfolioBias.LongShort, percent=0.03):
        if False:
            print('Hello World!')
        'Initialize a new instance of AccumulativeInsightPortfolioConstructionModel\n        Args:\n            rebalance: Rebalancing parameter. If it is a timedelta, date rules or Resolution, it will be converted into a function.\n                              If None will be ignored.\n                              The function returns the next expected rebalance time for a given algorithm UTC DateTime.\n                              The function returns null if unknown, in which case the function will be called again in the\n                              next loop. Returning current time will trigger rebalance.\n            portfolioBias: Specifies the bias of the portfolio (Short, Long/Short, Long)\n            percent: percent of portfolio to allocate to each position'
        super().__init__(rebalance)
        self.portfolioBias = portfolioBias
        self.percent = abs(percent)
        self.sign = lambda x: -1 if x < 0 else 1 if x > 0 else 0

    def DetermineTargetPercent(self, activeInsights):
        if False:
            while True:
                i = 10
        'Will determine the target percent for each insight\n        Args:\n            activeInsights: The active insights to generate a target for'
        percentPerSymbol = {}
        insights = sorted(self.Algorithm.Insights.GetActiveInsights(self.currentUtcTime), key=lambda insight: insight.GeneratedTimeUtc)
        for insight in insights:
            targetPercent = 0
            if insight.Symbol in percentPerSymbol:
                targetPercent = percentPerSymbol[insight.Symbol]
                if insight.Direction == InsightDirection.Flat:
                    if abs(targetPercent) < self.percent:
                        targetPercent = 0
                    else:
                        targetPercent += -self.percent if targetPercent > 0 else self.percent
            targetPercent += self.percent * insight.Direction
            if self.portfolioBias != PortfolioBias.LongShort and self.sign(targetPercent) != self.portfolioBias:
                targetPercent = 0
            percentPerSymbol[insight.Symbol] = targetPercent
        return dict(((insight, percentPerSymbol[insight.Symbol]) for insight in activeInsights))

    def CreateTargets(self, algorithm, insights):
        if False:
            return 10
        'Create portfolio targets from the specified insights\n        Args:\n            algorithm: The algorithm instance\n            insights: The insights to create portfolio targets from\n        Returns:\n            An enumerable of portfolio targets to be sent to the execution model'
        self.currentUtcTime = algorithm.UtcTime
        return super().CreateTargets(algorithm, insights)