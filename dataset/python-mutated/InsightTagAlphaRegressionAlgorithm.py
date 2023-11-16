from AlgorithmImports import *

class InsightTagAlphaRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.spy = Symbol.Create('SPY', SecurityType.Equity, Market.USA)
        self.fb = Symbol.Create('FB', SecurityType.Equity, Market.USA)
        self.ibm = Symbol.Create('IBM', SecurityType.Equity, Market.USA)
        self.SetUniverseSelection(ManualUniverseSelectionModel([self.spy, self.fb, self.ibm]))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.AddAlpha(OneTimeAlphaModel(self.spy))
        self.AddAlpha(OneTimeAlphaModel(self.fb))
        self.AddAlpha(OneTimeAlphaModel(self.ibm))
        self.InsightsGenerated += self.OnInsightsGeneratedVerifier
        self.symbols_with_generated_insights = []

    def OnInsightsGeneratedVerifier(self, algorithm: IAlgorithm, insightsCollection: GeneratedInsightsCollection) -> None:
        if False:
            while True:
                i = 10
        for insight in insightsCollection.Insights:
            if insight.Tag != OneTimeAlphaModel.GenerateInsightTag(insight.Symbol):
                raise Exception('Unexpected insight tag was emitted')
            self.symbols_with_generated_insights.append(insight.Symbol)

    def OnEndOfAlgorithm(self) -> None:
        if False:
            i = 10
            return i + 15
        if len(self.symbols_with_generated_insights) != 3:
            raise Exception('Unexpected number of symbols with generated insights')
        if not self.spy in self.symbols_with_generated_insights:
            raise Exception('SPY symbol was not found in symbols with generated insights')
        if not self.fb in self.symbols_with_generated_insights:
            raise Exception('FB symbol was not found in symbols with generated insights')
        if not self.ibm in self.symbols_with_generated_insights:
            raise Exception('IBM symbol was not found in symbols with generated insights')

class OneTimeAlphaModel(AlphaModel):

    def __init__(self, symbol):
        if False:
            print('Hello World!')
        self.symbol = symbol
        self.triggered = False

    def Update(self, algorithm, data):
        if False:
            for i in range(10):
                print('nop')
        insights = []
        if not self.triggered:
            self.triggered = True
            insights.append(Insight.Price(self.symbol, Resolution.Daily, 1, InsightDirection.Down, tag=OneTimeAlphaModel.GenerateInsightTag(self.symbol)))
        return insights

    @staticmethod
    def GenerateInsightTag(symbol: Symbol) -> str:
        if False:
            while True:
                i = 10
        return f'Insight generated for {symbol}'