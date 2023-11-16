from AlgorithmImports import *
from Portfolio.MeanVarianceOptimizationPortfolioConstructionModel import *

class MeanVarianceOptimizationFrameworkAlgorithm(QCAlgorithm):
    """Mean Variance Optimization algorithm."""

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.UniverseSettings.Resolution = Resolution.Minute
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.symbols = [Symbol.Create(x, SecurityType.Equity, Market.USA) for x in ['AIG', 'BAC', 'IBM', 'SPY']]
        self.SetUniverseSelection(CoarseFundamentalUniverseSelectionModel(self.coarseSelector))
        self.SetAlpha(HistoricalReturnsAlphaModel(resolution=Resolution.Daily))
        self.SetPortfolioConstruction(MeanVarianceOptimizationPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

    def coarseSelector(self, coarse):
        if False:
            return 10
        last = 3 if self.Time.day > 8 else len(self.symbols)
        return self.symbols[0:last]

    def OnOrderEvent(self, orderEvent):
        if False:
            print('Hello World!')
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(str(orderEvent))