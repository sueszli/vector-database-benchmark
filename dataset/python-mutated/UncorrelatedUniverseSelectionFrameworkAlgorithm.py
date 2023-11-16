from AlgorithmImports import *
from Selection.UncorrelatedUniverseSelectionModel import UncorrelatedUniverseSelectionModel

class UncorrelatedUniverseSelectionFrameworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2018, 1, 1)
        self.SetCash(1000000)
        benchmark = Symbol.Create('SPY', SecurityType.Equity, Market.USA)
        self.SetUniverseSelection(UncorrelatedUniverseSelectionModel(benchmark))
        self.SetAlpha(UncorrelatedUniverseSelectionAlphaModel())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())

class UncorrelatedUniverseSelectionAlphaModel(AlphaModel):
    """Uses ranking of intraday percentage difference between open price and close price to create magnitude and direction prediction for insights"""

    def __init__(self, numberOfStocks=10, predictionInterval=timedelta(1)):
        if False:
            for i in range(10):
                print('nop')
        self.predictionInterval = predictionInterval
        self.numberOfStocks = numberOfStocks

    def Update(self, algorithm, data):
        if False:
            i = 10
            return i + 15
        symbolsRet = dict()
        for kvp in algorithm.ActiveSecurities:
            security = kvp.Value
            if security.HasData:
                open = security.Open
                if open != 0:
                    symbolsRet[security.Symbol] = security.Close / open - 1
        symbolsRet = dict(sorted(symbolsRet.items(), key=lambda kvp: abs(kvp[1]), reverse=True)[:self.numberOfStocks])
        insights = []
        for (symbol, price_change) in symbolsRet.items():
            direction = InsightDirection.Up if price_change > 0 else InsightDirection.Down
            insights.append(Insight.Price(symbol, self.predictionInterval, direction, abs(price_change), None))
        return insights