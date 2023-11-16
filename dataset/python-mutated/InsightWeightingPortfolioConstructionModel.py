from AlgorithmImports import *
from EqualWeightingPortfolioConstructionModel import EqualWeightingPortfolioConstructionModel, PortfolioBias

class InsightWeightingPortfolioConstructionModel(EqualWeightingPortfolioConstructionModel):
    """Provides an implementation of IPortfolioConstructionModel that generates percent targets based on the
    Insight.Weight. The target percent holdings of each Symbol is given by the Insight.Weight from the last
    active Insight for that symbol.
    For insights of direction InsightDirection.Up, long targets are returned and for insights of direction
    InsightDirection.Down, short targets are returned.
    If the sum of all the last active Insight per symbol is bigger than 1, it will factor down each target
    percent holdings proportionally so the sum is 1.
    It will ignore Insight that have no Insight.Weight value."""

    def __init__(self, rebalance=Resolution.Daily, portfolioBias=PortfolioBias.LongShort):
        if False:
            print('Hello World!')
        'Initialize a new instance of InsightWeightingPortfolioConstructionModel\n        Args:\n            rebalance: Rebalancing parameter. If it is a timedelta, date rules or Resolution, it will be converted into a function.\n                              If None will be ignored.\n                              The function returns the next expected rebalance time for a given algorithm UTC DateTime.\n                              The function returns null if unknown, in which case the function will be called again in the\n                              next loop. Returning current time will trigger rebalance.\n            portfolioBias: Specifies the bias of the portfolio (Short, Long/Short, Long)'
        super().__init__(rebalance, portfolioBias)

    def ShouldCreateTargetForInsight(self, insight):
        if False:
            print('Hello World!')
        'Method that will determine if the portfolio construction model should create a\n        target for this insight\n        Args:\n            insight: The insight to create a target for'
        return insight.Weight is not None

    def DetermineTargetPercent(self, activeInsights):
        if False:
            i = 10
            return i + 15
        'Will determine the target percent for each insight\n        Args:\n            activeInsights: The active insights to generate a target for'
        result = {}
        weightSums = sum((self.GetValue(insight) for insight in activeInsights if self.RespectPortfolioBias(insight)))
        weightFactor = 1.0
        if weightSums > 1:
            weightFactor = 1 / weightSums
        for insight in activeInsights:
            result[insight] = (insight.Direction if self.RespectPortfolioBias(insight) else InsightDirection.Flat) * self.GetValue(insight) * weightFactor
        return result

    def GetValue(self, insight):
        if False:
            print('Hello World!')
        'Method that will determine which member will be used to compute the weights and gets its value\n        Args:\n            insight: The insight to create a target for\n        Returns:\n            The value of the selected insight member'
        return abs(insight.Weight)