from AlgorithmImports import *

class AddAlphaModelAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        ' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        spy = Symbol.Create('SPY', SecurityType.Equity, Market.USA)
        fb = Symbol.Create('FB', SecurityType.Equity, Market.USA)
        ibm = Symbol.Create('IBM', SecurityType.Equity, Market.USA)
        self.SetUniverseSelection(ManualUniverseSelectionModel([spy, fb, ibm]))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.AddAlpha(OneTimeAlphaModel(spy))
        self.AddAlpha(OneTimeAlphaModel(fb))
        self.AddAlpha(OneTimeAlphaModel(ibm))

class OneTimeAlphaModel(AlphaModel):

    def __init__(self, symbol):
        if False:
            return 10
        self.symbol = symbol
        self.triggered = False

    def Update(self, algorithm, data):
        if False:
            for i in range(10):
                print('nop')
        insights = []
        if not self.triggered:
            self.triggered = True
            insights.append(Insight.Price(self.symbol, Resolution.Daily, 1, InsightDirection.Down))
        return insights