from AlgorithmImports import *

class HistoricalReturnsAlphaModel(AlphaModel):
    """Uses Historical returns to create insights."""

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Initializes a new default instance of the HistoricalReturnsAlphaModel class.\n        Args:\n            lookback(int): Historical return lookback period\n            resolution: The resolution of historical data'
        self.lookback = kwargs['lookback'] if 'lookback' in kwargs else 1
        self.resolution = kwargs['resolution'] if 'resolution' in kwargs else Resolution.Daily
        self.predictionInterval = Time.Multiply(Extensions.ToTimeSpan(self.resolution), self.lookback)
        self.symbolDataBySymbol = {}
        self.insightCollection = InsightCollection()

    def Update(self, algorithm, data):
        if False:
            i = 10
            return i + 15
        'Updates this alpha model with the latest data from the algorithm.\n        This is called each time the algorithm receives data for subscribed securities\n        Args:\n            algorithm: The algorithm instance\n            data: The new data available\n        Returns:\n            The new insights generated'
        insights = []
        for (symbol, symbolData) in self.symbolDataBySymbol.items():
            if symbolData.CanEmit:
                direction = InsightDirection.Flat
                magnitude = symbolData.Return
                if magnitude > 0:
                    direction = InsightDirection.Up
                if magnitude < 0:
                    direction = InsightDirection.Down
                if direction == InsightDirection.Flat:
                    self.CancelInsights(algorithm, symbol)
                    continue
                insights.append(Insight.Price(symbol, self.predictionInterval, direction, magnitude, None))
        self.insightCollection.AddRange(insights)
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            while True:
                i = 10
        'Event fired each time the we add/remove securities from the data feed\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        for removed in changes.RemovedSecurities:
            symbolData = self.symbolDataBySymbol.pop(removed.Symbol, None)
            if symbolData is not None:
                symbolData.RemoveConsolidators(algorithm)
            self.CancelInsights(algorithm, removed.Symbol)
        symbols = [x.Symbol for x in changes.AddedSecurities]
        history = algorithm.History(symbols, self.lookback, self.resolution)
        if history.empty:
            return
        tickers = history.index.levels[0]
        for ticker in tickers:
            symbol = SymbolCache.GetSymbol(ticker)
            if symbol not in self.symbolDataBySymbol:
                symbolData = SymbolData(symbol, self.lookback)
                self.symbolDataBySymbol[symbol] = symbolData
                symbolData.RegisterIndicators(algorithm, self.resolution)
                symbolData.WarmUpIndicators(history.loc[ticker])

    def CancelInsights(self, algorithm, symbol):
        if False:
            return 10
        if not self.insightCollection.ContainsKey(symbol):
            return
        insights = self.insightCollection[symbol]
        algorithm.Insights.Cancel(insights)
        self.insightCollection.Clear([symbol])

class SymbolData:
    """Contains data specific to a symbol required by this model"""

    def __init__(self, symbol, lookback):
        if False:
            i = 10
            return i + 15
        self.Symbol = symbol
        self.ROC = RateOfChange('{}.ROC({})'.format(symbol, lookback), lookback)
        self.Consolidator = None
        self.previous = 0

    def RegisterIndicators(self, algorithm, resolution):
        if False:
            i = 10
            return i + 15
        self.Consolidator = algorithm.ResolveConsolidator(self.Symbol, resolution)
        algorithm.RegisterIndicator(self.Symbol, self.ROC, self.Consolidator)

    def RemoveConsolidators(self, algorithm):
        if False:
            while True:
                i = 10
        if self.Consolidator is not None:
            algorithm.SubscriptionManager.RemoveConsolidator(self.Symbol, self.Consolidator)

    def WarmUpIndicators(self, history):
        if False:
            print('Hello World!')
        for tuple in history.itertuples():
            self.ROC.Update(tuple.Index, tuple.close)

    @property
    def Return(self):
        if False:
            return 10
        return float(self.ROC.Current.Value)

    @property
    def CanEmit(self):
        if False:
            while True:
                i = 10
        if self.previous == self.ROC.Samples:
            return False
        self.previous = self.ROC.Samples
        return self.ROC.IsReady

    def __str__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return '{}: {:.2%}'.format(self.ROC.Name, (1 + self.Return) ** 252 - 1)