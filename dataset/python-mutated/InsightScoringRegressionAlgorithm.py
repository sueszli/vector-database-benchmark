from AlgorithmImports import *

class InsightScoringRegressionAlgorithm(QCAlgorithm):
    """Regression algorithm showing how to define a custom insight evaluator"""

    def Initialize(self):
        if False:
            while True:
                i = 10
        ' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        symbols = [Symbol.Create('SPY', SecurityType.Equity, Market.USA)]
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(minutes=20), 0.025, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel(Resolution.Daily))
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.01))
        self.Insights.SetInsightScoreFunction(CustomInsightScoreFunction(self.Securities))

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        allInsights = self.Insights.GetInsights(lambda insight: True)
        if len(allInsights) != 100 or len(self.Insights.GetInsights()) != 100:
            raise ValueError(f'Unexpected insight count found {allInsights.Count}')
        if sum((1 for insight in allInsights if insight.Score.Magnitude == 0 or insight.Score.Direction == 0)) < 5:
            raise ValueError(f'Insights not scored!')
        if sum((1 for insight in allInsights if insight.Score.IsFinalScore)) < 99:
            raise ValueError(f'Insights not finalized!')

class CustomInsightScoreFunction:

    def __init__(self, securities):
        if False:
            i = 10
            return i + 15
        self._securities = securities
        self._openInsights = {}

    def Score(self, insightManager, utcTime):
        if False:
            print('Hello World!')
        openInsights = insightManager.GetActiveInsights(utcTime)
        for insight in openInsights:
            self._openInsights[insight.Id] = insight
        toRemove = []
        for openInsight in self._openInsights.values():
            security = self._securities[openInsight.Symbol]
            openInsight.ReferenceValueFinal = security.Price
            score = openInsight.ReferenceValueFinal - openInsight.ReferenceValue
            openInsight.Score.SetScore(InsightScoreType.Direction, score, utcTime)
            openInsight.Score.SetScore(InsightScoreType.Magnitude, score * 2, utcTime)
            openInsight.EstimatedValue = score * 100
            if openInsight.IsExpired(utcTime):
                openInsight.Score.Finalize(utcTime)
                toRemove.append(openInsight)
        for insightToRemove in toRemove:
            self._openInsights.pop(insightToRemove.Id)