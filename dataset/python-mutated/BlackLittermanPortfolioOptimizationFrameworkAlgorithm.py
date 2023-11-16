from AlgorithmImports import *
from Alphas.HistoricalReturnsAlphaModel import HistoricalReturnsAlphaModel
from Portfolio.BlackLittermanOptimizationPortfolioConstructionModel import *
from Portfolio.UnconstrainedMeanVariancePortfolioOptimizer import UnconstrainedMeanVariancePortfolioOptimizer
from Risk.NullRiskManagementModel import NullRiskManagementModel

class BlackLittermanPortfolioOptimizationFrameworkAlgorithm(QCAlgorithm):
    """Black-Litterman Optimization algorithm."""

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.symbols = [Symbol.Create(x, SecurityType.Equity, Market.USA) for x in ['AIG', 'BAC', 'IBM', 'SPY']]
        optimizer = UnconstrainedMeanVariancePortfolioOptimizer()
        self.SetUniverseSelection(CoarseFundamentalUniverseSelectionModel(self.coarseSelector))
        self.SetAlpha(HistoricalReturnsAlphaModel(resolution=Resolution.Daily))
        self.SetPortfolioConstruction(BlackLittermanOptimizationPortfolioConstructionModel(optimizer=optimizer))
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

    def coarseSelector(self, coarse):
        if False:
            return 10
        last = 3 if self.Time.day > 8 else len(self.symbols)
        return self.symbols[0:last]

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(orderEvent)