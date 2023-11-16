from AlgorithmImports import *
from InsightWeightingPortfolioConstructionModel import InsightWeightingPortfolioConstructionModel

class ConfidenceWeightedPortfolioConstructionModel(InsightWeightingPortfolioConstructionModel):
    """Provides an implementation of IPortfolioConstructionModel that generates percent targets based on the
    Insight.Confidence. The target percent holdings of each Symbol is given by the Insight.Confidence from the last
    active Insight for that symbol.
    For insights of direction InsightDirection.Up, long targets are returned and for insights of direction
    InsightDirection.Down, short targets are returned.
    If the sum of all the last active Insight per symbol is bigger than 1, it will factor down each target
    percent holdings proportionally so the sum is 1.
    It will ignore Insight that have no Insight.Confidence value."""

    def __init__(self, rebalance=Resolution.Daily, portfolioBias=PortfolioBias.LongShort):
        if False:
            return 10
        'Initialize a new instance of ConfidenceWeightedPortfolioConstructionModel\n        Args:\n            rebalance: Rebalancing parameter. If it is a timedelta, date rules or Resolution, it will be converted into a function.\n                              If None will be ignored.\n                              The function returns the next expected rebalance time for a given algorithm UTC DateTime.\n                              The function returns null if unknown, in which case the function will be called again in the\n                              next loop. Returning current time will trigger rebalance.\n            portfolioBias: Specifies the bias of the portfolio (Short, Long/Short, Long)'
        super().__init__(rebalance, portfolioBias)

    def ShouldCreateTargetForInsight(self, insight):
        if False:
            while True:
                i = 10
        'Method that will determine if the portfolio construction model should create a\n        target for this insight\n        Args:\n            insight: The insight to create a target for'
        return insight.Confidence is not None

    def GetValue(self, insight):
        if False:
            while True:
                i = 10
        'Method that will determine which member will be used to compute the weights and gets its value\n        Args:\n            insight: The insight to create a target for\n        Returns:\n            The value of the selected insight member'
        return insight.Confidence