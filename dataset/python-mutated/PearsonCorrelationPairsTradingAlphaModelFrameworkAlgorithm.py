from AlgorithmImports import *

class PearsonCorrelationPairsTradingAlphaModelFrameworkAlgorithm(QCAlgorithm):
    """Framework algorithm that uses the PearsonCorrelationPairsTradingAlphaModel.
    This model extendes BasePairsTradingAlphaModel and uses Pearson correlation
    to rank the pairs trading candidates and use the best candidate to trade."""

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        symbols = [Symbol.Create(ticker, SecurityType.Equity, Market.USA) for ticker in ['SPY', 'AIG', 'BAC', 'IBM']]
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols[:2]))
        self.AddUniverseSelection(ScheduledUniverseSelectionModel(self.DateRules.EveryDay(), self.TimeRules.Midnight, lambda dt: symbols if dt.day <= (self.EndDate - timedelta(1)).day else []))
        self.SetAlpha(PearsonCorrelationPairsTradingAlphaModel(252, Resolution.Daily))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

    def OnEndOfAlgorithm(self) -> None:
        if False:
            return 10
        consolidatorCount = sum((s.Consolidators.Count for s in self.SubscriptionManager.Subscriptions))
        if consolidatorCount > 0:
            raise Exception(f'The number of consolidator should be zero. Actual: {consolidatorCount}')