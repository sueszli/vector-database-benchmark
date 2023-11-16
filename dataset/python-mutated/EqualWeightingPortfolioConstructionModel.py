from AlgorithmImports import *

class EqualWeightingPortfolioConstructionModel(PortfolioConstructionModel):
    """Provides an implementation of IPortfolioConstructionModel that gives equal weighting to all securities.
    The target percent holdings of each security is 1/N where N is the number of securities.
    For insights of direction InsightDirection.Up, long targets are returned and
    for insights of direction InsightDirection.Down, short targets are returned."""

    def __init__(self, rebalance=Resolution.Daily, portfolioBias=PortfolioBias.LongShort):
        if False:
            while True:
                i = 10
        'Initialize a new instance of EqualWeightingPortfolioConstructionModel\n        Args:\n            rebalance: Rebalancing parameter. If it is a timedelta, date rules or Resolution, it will be converted into a function.\n                              If None will be ignored.\n                              The function returns the next expected rebalance time for a given algorithm UTC DateTime.\n                              The function returns null if unknown, in which case the function will be called again in the\n                              next loop. Returning current time will trigger rebalance.\n            portfolioBias: Specifies the bias of the portfolio (Short, Long/Short, Long)'
        super().__init__()
        self.portfolioBias = portfolioBias
        rebalancingFunc = rebalance
        if isinstance(rebalance, int):
            rebalance = Extensions.ToTimeSpan(rebalance)
        if isinstance(rebalance, timedelta):
            rebalancingFunc = lambda dt: dt + rebalance
        if rebalancingFunc:
            self.SetRebalancingFunc(rebalancingFunc)

    def DetermineTargetPercent(self, activeInsights):
        if False:
            return 10
        'Will determine the target percent for each insight\n        Args:\n            activeInsights: The active insights to generate a target for'
        result = {}
        count = sum((x.Direction != InsightDirection.Flat and self.RespectPortfolioBias(x) for x in activeInsights))
        percent = 0 if count == 0 else 1.0 / count
        for insight in activeInsights:
            result[insight] = (insight.Direction if self.RespectPortfolioBias(insight) else InsightDirection.Flat) * percent
        return result

    def RespectPortfolioBias(self, insight):
        if False:
            while True:
                i = 10
        'Method that will determine if a given insight respects the portfolio bias\n        Args:\n            insight: The insight to create a target for\n        '
        return self.portfolioBias == PortfolioBias.LongShort or insight.Direction == self.portfolioBias