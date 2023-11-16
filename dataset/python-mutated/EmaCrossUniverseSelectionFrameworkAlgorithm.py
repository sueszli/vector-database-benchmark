from AlgorithmImports import *
from Alphas.ConstantAlphaModel import ConstantAlphaModel
from Selection.EmaCrossUniverseSelectionModel import EmaCrossUniverseSelectionModel
from Portfolio.EqualWeightingPortfolioConstructionModel import EqualWeightingPortfolioConstructionModel

class EmaCrossUniverseSelectionFrameworkAlgorithm(QCAlgorithm):
    """Framework algorithm that uses the EmaCrossUniverseSelectionModel to select the universe based on a moving average cross."""

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 1, 1)
        self.SetEndDate(2015, 1, 1)
        self.SetCash(100000)
        fastPeriod = 100
        slowPeriod = 300
        count = 10
        self.UniverseSettings.Leverage = 2.0
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetUniverseSelection(EmaCrossUniverseSelectionModel(fastPeriod, slowPeriod, count))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(1), None, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())