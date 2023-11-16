from AlgorithmImports import *

class AddRiskManagementAlgorithm(QCAlgorithm):
    """Basic template framework algorithm uses framework components to define the algorithm."""

    def Initialize(self):
        if False:
            return 10
        ' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        symbols = [Symbol.Create('SPY', SecurityType.Equity, Market.USA)]
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(minutes=20), 0.025, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        riskModel = CompositeRiskManagementModel(MaximumDrawdownPercentPortfolio(0.02))
        riskModel.AddRiskManagement(MaximumUnrealizedProfitPercentPerSecurity(0.01))
        self.SetRiskManagement(MaximumDrawdownPercentPortfolio(0.02))
        self.AddRiskManagement(MaximumUnrealizedProfitPercentPerSecurity(0.01))