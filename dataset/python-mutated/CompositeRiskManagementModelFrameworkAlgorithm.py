from AlgorithmImports import *
from Risk.CompositeRiskManagementModel import CompositeRiskManagementModel
from Risk.MaximumUnrealizedProfitPercentPerSecurity import MaximumUnrealizedProfitPercentPerSecurity
from Risk.MaximumDrawdownPercentPerSecurity import MaximumDrawdownPercentPerSecurity

class CompositeRiskManagementModelFrameworkAlgorithm(QCAlgorithm):
    """Show cases how to use the CompositeRiskManagementModel."""

    def Initialize(self):
        if False:
            print('Hello World!')
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.SetUniverseSelection(ManualUniverseSelectionModel([Symbol.Create('SPY', SecurityType.Equity, Market.USA)]))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(minutes=20), 0.025, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(CompositeRiskManagementModel(MaximumUnrealizedProfitPercentPerSecurity(0.01), MaximumDrawdownPercentPerSecurity(0.01)))