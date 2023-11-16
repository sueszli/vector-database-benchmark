from AlgorithmImports import *
from Alphas.RsiAlphaModel import RsiAlphaModel
from Portfolio.EqualWeightingPortfolioConstructionModel import EqualWeightingPortfolioConstructionModel
from Execution.StandardDeviationExecutionModel import StandardDeviationExecutionModel

class StandardDeviationExecutionModelRegressionAlgorithm(QCAlgorithm):
    """Regression algorithm for the StandardDeviationExecutionModel.
    This algorithm shows how the execution model works to split up orders and submit them
    only when the price is 2 standard deviations from the 60min mean (default model settings)."""

    def Initialize(self):
        if False:
            print('Hello World!')
        ' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(1000000)
        self.SetUniverseSelection(ManualUniverseSelectionModel([Symbol.Create('AIG', SecurityType.Equity, Market.USA), Symbol.Create('BAC', SecurityType.Equity, Market.USA), Symbol.Create('IBM', SecurityType.Equity, Market.USA), Symbol.Create('SPY', SecurityType.Equity, Market.USA)]))
        self.SetAlpha(RsiAlphaModel(14, Resolution.Hour))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(StandardDeviationExecutionModel())

    def OnOrderEvent(self, orderEvent):
        if False:
            while True:
                i = 10
        self.Log(f'{self.Time}: {orderEvent}')