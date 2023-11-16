from AlgorithmImports import *

class BaseFrameworkRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2014, 6, 1)
        self.SetEndDate(2014, 6, 30)
        self.UniverseSettings.Resolution = Resolution.Hour
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Raw
        symbols = [Symbol.Create(ticker, SecurityType.Equity, Market.USA) for ticker in ['AAPL', 'AIG', 'BAC', 'SPY']]
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols[:2]))
        self.AddUniverseSelection(ScheduledUniverseSelectionModel(self.DateRules.EveryDay(), self.TimeRules.Midnight, lambda dt: symbols if dt.replace(tzinfo=None) < self.EndDate - timedelta(1) else []))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(31), 0.025, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        insightsCount = len(self.Insights.GetInsights(lambda insight: insight.IsActive(self.UtcTime)))
        if insightsCount != 0:
            raise Exception(f'The number of active insights should be 0. Actual: {insightsCount}')