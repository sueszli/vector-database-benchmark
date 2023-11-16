from AlgorithmImports import *

class TripleLeverageETFPairVolatilityDecayAlpha(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2018, 1, 1)
        self.SetCash(100000)
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))
        ultraLong = Symbol.Create('UGLD', SecurityType.Equity, Market.USA)
        ultraShort = Symbol.Create('DGLD', SecurityType.Equity, Market.USA)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetUniverseSelection(ManualUniverseSelectionModel([ultraLong, ultraShort]))
        self.SetAlpha(RebalancingTripleLeveragedETFAlphaModel(ultraLong, ultraShort))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

class RebalancingTripleLeveragedETFAlphaModel(AlphaModel):
    """
        Rebalance a pair of 3x leveraged ETFs and predict that the value of both ETFs in each pair will decrease.
    """

    def __init__(self, ultraLong, ultraShort):
        if False:
            while True:
                i = 10
        self.period = timedelta(1)
        self.magnitude = 0.001
        self.ultraLong = ultraLong
        self.ultraShort = ultraShort
        self.Name = 'RebalancingTripleLeveragedETFAlphaModel'

    def Update(self, algorithm, data):
        if False:
            print('Hello World!')
        return Insight.Group([Insight.Price(self.ultraLong, self.period, InsightDirection.Down, self.magnitude), Insight.Price(self.ultraShort, self.period, InsightDirection.Down, self.magnitude)])