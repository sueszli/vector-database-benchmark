from AlgorithmImports import *

class ExpiryHelperAlphaModelFrameworkAlgorithm(QCAlgorithm):
    """Expiry Helper framework algorithm uses Expiry helper class in an Alpha Model"""

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        ' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.UniverseSettings.Resolution = Resolution.Hour
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2014, 1, 1)
        self.SetCash(100000)
        symbols = [Symbol.Create('SPY', SecurityType.Equity, Market.USA)]
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.SetAlpha(self.ExpiryHelperAlphaModel())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.01))
        self.InsightsGenerated += self.OnInsightsGenerated

    def OnInsightsGenerated(self, s, e):
        if False:
            for i in range(10):
                print('nop')
        for insight in e.Insights:
            self.Log(f'{e.DateTimeUtc.isoweekday()}: Close Time {insight.CloseTimeUtc} {insight.CloseTimeUtc.isoweekday()}')

    class ExpiryHelperAlphaModel(AlphaModel):
        nextUpdate = None
        direction = InsightDirection.Up

        def Update(self, algorithm, data):
            if False:
                while True:
                    i = 10
            if self.nextUpdate is not None and self.nextUpdate > algorithm.Time:
                return []
            expiry = Expiry.EndOfDay
            self.nextUpdate = expiry(algorithm.Time)
            weekday = algorithm.Time.isoweekday()
            insights = []
            for symbol in data.Bars.Keys:
                if weekday == 1:
                    insights.append(Insight.Price(symbol, Expiry.OneMonth, self.direction))
                elif weekday == 2:
                    insights.append(Insight.Price(symbol, Expiry.EndOfMonth, self.direction))
                elif weekday == 3:
                    insights.append(Insight.Price(symbol, Expiry.EndOfWeek, self.direction))
                elif weekday == 4:
                    insights.append(Insight.Price(symbol, Expiry.EndOfDay, self.direction))
            return insights