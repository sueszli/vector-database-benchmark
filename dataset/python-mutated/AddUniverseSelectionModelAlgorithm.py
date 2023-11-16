from AlgorithmImports import *

class AddUniverseSelectionModelAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        ' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(minutes=20), 0.025, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetUniverseSelection(ManualUniverseSelectionModel([Symbol.Create('SPY', SecurityType.Equity, Market.USA)]))
        self.AddUniverseSelection(ManualUniverseSelectionModel([Symbol.Create('AAPL', SecurityType.Equity, Market.USA)]))
        self.AddUniverseSelection(ManualUniverseSelectionModel(Symbol.Create('SPY', SecurityType.Equity, Market.USA), Symbol.Create('FB', SecurityType.Equity, Market.USA)))

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        if self.UniverseManager.Count != 3:
            raise ValueError('Unexpected universe count')
        if self.UniverseManager.ActiveSecurities.Count != 3:
            raise ValueError('Unexpected active securities')