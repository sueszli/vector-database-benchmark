from AlgorithmImports import *

class IntradayReversalCurrencyMarketsAlpha(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2015, 1, 1)
        self.SetCash(100000)
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))
        resolution = Resolution.Hour
        symbols = [Symbol.Create('EURUSD', SecurityType.Forex, Market.Oanda)]
        self.UniverseSettings.Resolution = resolution
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.SetAlpha(IntradayReversalAlphaModel(5, resolution))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())
        self.SetWarmUp(20)

class IntradayReversalAlphaModel(AlphaModel):
    """Alpha model that uses a Price/SMA Crossover to create insights on Hourly Frequency.
    Frequency: Hourly data with 5-hour simple moving average.
    Strategy:
    Reversal strategy that goes Long when price crosses below SMA and Short when price crosses above SMA.
    The trading strategy is implemented only between 10AM - 3PM (NY time)"""

    def __init__(self, period_sma=5, resolution=Resolution.Hour):
        if False:
            while True:
                i = 10
        self.period_sma = period_sma
        self.resolution = resolution
        self.cache = {}
        self.Name = 'IntradayReversalAlphaModel'

    def Update(self, algorithm, data):
        if False:
            return 10
        timeToClose = algorithm.Time.replace(hour=15, minute=1, second=0)
        insights = []
        for kvp in algorithm.ActiveSecurities:
            symbol = kvp.Key
            if self.ShouldEmitInsight(algorithm, symbol) and symbol in self.cache:
                price = kvp.Value.Price
                symbolData = self.cache[symbol]
                direction = InsightDirection.Up if symbolData.is_uptrend(price) else InsightDirection.Down
                if direction == symbolData.PreviousDirection:
                    continue
                symbolData.PreviousDirection = direction
                insights.append(Insight.Price(symbol, timeToClose, direction))
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            print('Hello World!')
        'Handle creation of the new security and its cache class.\n        Simplified in this example as there is 1 asset.'
        for security in changes.AddedSecurities:
            self.cache[security.Symbol] = SymbolData(algorithm, security.Symbol, self.period_sma, self.resolution)

    def ShouldEmitInsight(self, algorithm, symbol):
        if False:
            return 10
        'Time to control when to start and finish emitting (10AM to 3PM)'
        timeOfDay = algorithm.Time.time()
        return algorithm.Securities[symbol].HasData and timeOfDay >= time(10) and (timeOfDay <= time(15))

class SymbolData:

    def __init__(self, algorithm, symbol, period_sma, resolution):
        if False:
            return 10
        self.PreviousDirection = InsightDirection.Flat
        self.priceSMA = algorithm.SMA(symbol, period_sma, resolution)

    def is_uptrend(self, price):
        if False:
            for i in range(10):
                print('nop')
        return self.priceSMA.IsReady and price < round(self.priceSMA.Current.Value * 1.001, 6)