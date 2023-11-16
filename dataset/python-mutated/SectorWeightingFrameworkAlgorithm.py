from AlgorithmImports import *
from Portfolio.SectorWeightingPortfolioConstructionModel import SectorWeightingPortfolioConstructionModel

class SectorWeightingFrameworkAlgorithm(QCAlgorithm):
    """This example algorithm defines its own custom coarse/fine fundamental selection model
    with sector weighted portfolio."""

    def Initialize(self):
        if False:
            print('Hello World!')
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2014, 4, 2)
        self.SetEndDate(2014, 4, 6)
        self.SetCash(100000)
        self.SetUniverseSelection(FineFundamentalUniverseSelectionModel(self.SelectCoarse, self.SelectFine))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(1)))
        self.SetPortfolioConstruction(SectorWeightingPortfolioConstructionModel())

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f'Order event: {orderEvent}. Holding value: {self.Securities[orderEvent.Symbol].Holdings.AbsoluteHoldingsValue}')

    def SelectCoarse(self, coarse):
        if False:
            i = 10
            return i + 15
        tickers = ['AAPL', 'AIG', 'IBM'] if self.Time.date() < date(2014, 4, 4) else ['GOOG', 'BAC', 'SPY']
        return [Symbol.Create(x, SecurityType.Equity, Market.USA) for x in tickers]

    def SelectFine(self, fine):
        if False:
            print('Hello World!')
        return [f.Symbol for f in fine]