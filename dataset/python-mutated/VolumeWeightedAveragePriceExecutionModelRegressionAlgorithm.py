from AlgorithmImports import *
from Alphas.RsiAlphaModel import RsiAlphaModel
from Portfolio.EqualWeightingPortfolioConstructionModel import EqualWeightingPortfolioConstructionModel
from Execution.VolumeWeightedAveragePriceExecutionModel import VolumeWeightedAveragePriceExecutionModel

class VolumeWeightedAveragePriceExecutionModelRegressionAlgorithm(QCAlgorithm):
    """Regression algorithm for the VolumeWeightedAveragePriceExecutionModel.
    This algorithm shows how the execution model works to split up orders and
    submit them only when the price is on the favorable side of the intraday VWAP."""

    def Initialize(self):
        if False:
            print('Hello World!')
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(1000000)
        self.SetUniverseSelection(ManualUniverseSelectionModel([Symbol.Create('AIG', SecurityType.Equity, Market.USA), Symbol.Create('BAC', SecurityType.Equity, Market.USA), Symbol.Create('IBM', SecurityType.Equity, Market.USA), Symbol.Create('SPY', SecurityType.Equity, Market.USA)]))
        self.SetAlpha(RsiAlphaModel(14, Resolution.Hour))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(VolumeWeightedAveragePriceExecutionModel())
        self.InsightsGenerated += self.OnInsightsGenerated

    def OnInsightsGenerated(self, algorithm, data):
        if False:
            for i in range(10):
                print('nop')
        self.Log(f"{self.Time}: {', '.join((str(x) for x in data.Insights))}")

    def OnOrderEvent(self, orderEvent):
        if False:
            for i in range(10):
                print('nop')
        self.Log(f'{self.Time}: {orderEvent}')