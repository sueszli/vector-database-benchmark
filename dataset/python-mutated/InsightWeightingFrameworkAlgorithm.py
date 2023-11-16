from AlgorithmImports import *

class InsightWeightingFrameworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        ' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        symbols = [Symbol.Create('SPY', SecurityType.Equity, Market.USA)]
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(minutes=20), 0.025, None, 0.25))
        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        if self.Portfolio.TotalHoldingsValue > self.Portfolio.TotalPortfolioValue * 0.28 or self.Portfolio.TotalHoldingsValue < self.Portfolio.TotalPortfolioValue * 0.23:
            raise ValueError('Unexpected Total Holdings Value: ' + str(self.Portfolio.TotalHoldingsValue))