from AlgorithmImports import *

class PriceGapMeanReversionAlpha(QCAlgorithm):
    """The motivating idea for this Alpha Model is that a large price gap (here we use true outliers --
    price gaps that whose absolutely values are greater than 3 * Volatility) is due to rebound
    back to an appropriate price or at least retreat from its brief extreme. Using a Coarse Universe selection
    function, the algorithm selects the top x-companies by Dollar Volume (x can be any number you choose)
    to trade with, and then uses the Standard Deviation of the 100 most-recent closing prices to determine
    which price movements are outliers that warrant emitting insights.

    This alpha is part of the Benchmark Alpha Series created by QuantConnect which are open
    sourced so the community and client funds can see an example of an alpha."""

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2018, 1, 1)
        self.SetCash(100000)
        self.week = -1
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetUniverseSelection(CoarseFundamentalUniverseSelectionModel(self.CoarseSelectionFunction))
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))
        self.SetAlpha(PriceGapMeanReversionAlphaModel())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

    def CoarseSelectionFunction(self, coarse):
        if False:
            return 10
        current_week = self.Time.isocalendar()[1]
        if current_week == self.week:
            return Universe.Unchanged
        self.week = current_week
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sortedByDollarVolume[:25]]

class PriceGapMeanReversionAlphaModel:

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        " Initialize variables and dictionary for Symbol Data to support algorithm's function "
        self.lookback = 100
        self.resolution = kwargs['resolution'] if 'resolution' in kwargs else Resolution.Minute
        self.prediction_interval = Time.Multiply(Extensions.ToTimeSpan(self.resolution), 5)
        self.symbolDataBySymbol = {}

    def Update(self, algorithm, data):
        if False:
            return 10
        insights = []
        for (symbol, symbolData) in self.symbolDataBySymbol.items():
            if not symbolData.IsTrend(data):
                continue
            direction = InsightDirection.Down if symbolData.PriceJump > 0 else InsightDirection.Up
            insights.append(Insight.Price(symbol, self.prediction_interval, direction, symbolData.PriceJump, None))
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            while True:
                i = 10
        for removed in changes.RemovedSecurities:
            symbolData = self.symbolDataBySymbol.pop(removed.Symbol, None)
            if symbolData is not None:
                symbolData.RemoveConsolidators(algorithm)
        symbols = [x.Symbol for x in changes.AddedSecurities if x.Symbol not in self.symbolDataBySymbol]
        history = algorithm.History(symbols, self.lookback, self.resolution)
        if history.empty:
            return
        for symbol in symbols:
            symbolData = SymbolData(algorithm, symbol, self.lookback, self.resolution)
            symbolData.WarmUpIndicators(history.loc[symbol])
            self.symbolDataBySymbol[symbol] = symbolData

class SymbolData:

    def __init__(self, algorithm, symbol, lookback, resolution):
        if False:
            return 10
        self.symbol = symbol
        self.close = 0
        self.last_price = 0
        self.PriceJump = 0
        self.consolidator = algorithm.ResolveConsolidator(symbol, resolution)
        self.volatility = StandardDeviation(f'{symbol}.STD({lookback})', lookback)
        algorithm.RegisterIndicator(symbol, self.volatility, self.consolidator)

    def RemoveConsolidators(self, algorithm):
        if False:
            return 10
        algorithm.SubscriptionManager.RemoveConsolidator(self.symbol, self.consolidator)

    def WarmUpIndicators(self, history):
        if False:
            print('Hello World!')
        self.close = history.iloc[-1].close
        for tuple in history.itertuples():
            self.volatility.Update(tuple.Index, tuple.close)

    def IsTrend(self, data):
        if False:
            for i in range(10):
                print('nop')
        if not data.Bars.ContainsKey(self.symbol):
            return False
        self.last_price = self.close
        self.close = data.Bars[self.symbol].Close
        self.PriceJump = self.close / self.last_price - 1
        return abs(100 * self.PriceJump) > 3 * self.volatility.Current.Value