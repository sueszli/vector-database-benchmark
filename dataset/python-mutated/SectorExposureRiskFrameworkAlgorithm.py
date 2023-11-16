from AlgorithmImports import *
from Portfolio.EqualWeightingPortfolioConstructionModel import EqualWeightingPortfolioConstructionModel
from Alphas.ConstantAlphaModel import ConstantAlphaModel
from Execution.ImmediateExecutionModel import ImmediateExecutionModel
from Risk.MaximumSectorExposureRiskManagementModel import MaximumSectorExposureRiskManagementModel

class SectorExposureRiskFrameworkAlgorithm(QCAlgorithm):
    """This example algorithm defines its own custom coarse/fine fundamental selection model
### with equally weighted portfolio and a maximum sector exposure."""

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2014, 3, 25)
        self.SetEndDate(2014, 4, 7)
        self.SetCash(100000)
        self.SetUniverseSelection(FineFundamentalUniverseSelectionModel(self.SelectCoarse, self.SelectFine))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(1)))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetRiskManagement(MaximumSectorExposureRiskManagementModel())

    def OnOrderEvent(self, orderEvent):
        if False:
            i = 10
            return i + 15
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f'Order event: {orderEvent}. Holding value: {self.Securities[orderEvent.Symbol].Holdings.AbsoluteHoldingsValue}')

    def SelectCoarse(self, coarse):
        if False:
            i = 10
            return i + 15
        tickers = ['AAPL', 'AIG', 'IBM'] if self.Time.date() < date(2014, 4, 1) else ['GOOG', 'BAC', 'SPY']
        return [Symbol.Create(x, SecurityType.Equity, Market.USA) for x in tickers]

    def SelectFine(self, fine):
        if False:
            while True:
                i = 10
        return [f.Symbol for f in fine]