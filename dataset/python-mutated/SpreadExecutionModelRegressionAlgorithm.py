from AlgorithmImports import *
from Alphas.RsiAlphaModel import RsiAlphaModel
from Portfolio.EqualWeightingPortfolioConstructionModel import EqualWeightingPortfolioConstructionModel
from Execution.SpreadExecutionModel import SpreadExecutionModel

class SpreadExecutionModelRegressionAlgorithm(QCAlgorithm):
    """Regression algorithm for the SpreadExecutionModel.
    This algorithm shows how the execution model works to 
    submit orders only when the price is on desirably tight spread."""

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetUniverseSelection(ManualUniverseSelectionModel([Symbol.Create('AIG', SecurityType.Equity, Market.USA), Symbol.Create('BAC', SecurityType.Equity, Market.USA), Symbol.Create('IBM', SecurityType.Equity, Market.USA), Symbol.Create('SPY', SecurityType.Equity, Market.USA)]))
        self.SetAlpha(RsiAlphaModel(14, Resolution.Hour))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(SpreadExecutionModel())
        self.InsightsGenerated += self.OnInsightsGenerated

    def OnInsightsGenerated(self, algorithm, data):
        if False:
            i = 10
            return i + 15
        self.Log(f"{self.Time}: {', '.join((str(x) for x in data.Insights))}")

    def OnOrderEvent(self, orderEvent):
        if False:
            for i in range(10):
                print('nop')
        self.Log(f'{self.Time}: {orderEvent}')