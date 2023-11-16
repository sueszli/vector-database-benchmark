from AlgorithmImports import *

class MeanReversionLunchBreakAlpha(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2018, 1, 1)
        self.SetCash(100000)
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))
        self.UniverseSettings.Resolution = Resolution.Hour
        self.SetUniverseSelection(CoarseFundamentalUniverseSelectionModel(self.CoarseSelectionFunction))
        self.SetAlpha(MeanReversionLunchBreakAlphaModel())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

    def CoarseSelectionFunction(self, coarse):
        if False:
            while True:
                i = 10
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        filtered = [x.Symbol for x in sortedByDollarVolume if not x.HasFundamentalData]
        return filtered[:20]

class MeanReversionLunchBreakAlphaModel(AlphaModel):
    """Uses the price return between the close of previous day to 12:00 the day after to
    predict mean-reversion of stock price during lunch break and creates direction prediction
    for insights accordingly."""

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        lookback = kwargs['lookback'] if 'lookback' in kwargs else 1
        self.resolution = Resolution.Hour
        self.predictionInterval = Time.Multiply(Extensions.ToTimeSpan(self.resolution), lookback)
        self.symbolDataBySymbol = dict()

    def Update(self, algorithm, data):
        if False:
            while True:
                i = 10
        for (symbol, symbolData) in self.symbolDataBySymbol.items():
            if data.Bars.ContainsKey(symbol):
                bar = data.Bars.GetValue(symbol)
                symbolData.Update(bar.EndTime, bar.Close)
        return [] if algorithm.Time.hour != 12 else [x.Insight for x in self.symbolDataBySymbol.values()]

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            i = 10
            return i + 15
        for security in changes.RemovedSecurities:
            self.symbolDataBySymbol.pop(security.Symbol, None)
        symbols = [x.Symbol for x in changes.AddedSecurities]
        history = algorithm.History(symbols, 1, self.resolution)
        if history.empty:
            algorithm.Debug(f'No data on {algorithm.Time}')
            return
        history = history.close.unstack(level=0)
        for (ticker, values) in history.iteritems():
            symbol = next((x for x in symbols if str(x) == ticker), None)
            if symbol in self.symbolDataBySymbol or symbol is None:
                continue
            self.symbolDataBySymbol[symbol] = self.SymbolData(symbol, self.predictionInterval)
            self.symbolDataBySymbol[symbol].Update(values.index[0], values[0])

    class SymbolData:

        def __init__(self, symbol, period):
            if False:
                print('Hello World!')
            self.symbol = symbol
            self.period = period
            self.meanOfPriceChange = IndicatorExtensions.SMA(RateOfChangePercent(1), 3)
            self.priceChange = RateOfChangePercent(3)

        def Update(self, time, value):
            if False:
                for i in range(10):
                    print('nop')
            return self.meanOfPriceChange.Update(time, value) and self.priceChange.Update(time, value)

        @property
        def Insight(self):
            if False:
                return 10
            direction = InsightDirection.Down if self.priceChange.Current.Value > 0 else InsightDirection.Up
            margnitude = abs(self.meanOfPriceChange.Current.Value)
            return Insight.Price(self.symbol, self.period, direction, margnitude, None)