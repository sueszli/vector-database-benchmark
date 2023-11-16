from AlgorithmImports import *
from Selection.ETFConstituentsUniverseSelectionModel import *

class ETFConstituentsFrameworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2020, 12, 1)
        self.SetEndDate(2020, 12, 7)
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        symbol = Symbol.Create('SPY', SecurityType.Equity, Market.USA)
        self.AddUniverseSelection(ETFConstituentsUniverseSelectionModel(symbol, self.UniverseSettings, self.ETFConstituentsFilter))
        self.AddAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(days=1)))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())

    def ETFConstituentsFilter(self, constituents: List[ETFConstituentData]) -> List[Symbol]:
        if False:
            i = 10
            return i + 15
        selected = sorted([c for c in constituents if c.Weight], key=lambda c: c.Weight, reverse=True)[:8]
        return [c.Symbol for c in selected]