from AlgorithmImports import *

class ShareClassMeanReversionAlpha(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2019, 1, 1)
        self.SetCash(100000)
        self.SetWarmUp(20)
        tickers = ['VIA', 'VIAB']
        self.UniverseSettings.Resolution = Resolution.Minute
        symbols = [Symbol.Create(ticker, SecurityType.Equity, Market.USA) for ticker in tickers]
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.SetAlpha(ShareClassMeanReversionAlphaModel(tickers=tickers))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

class ShareClassMeanReversionAlphaModel(AlphaModel):
    """ Initialize helper variables for the algorithm"""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.sma = SimpleMovingAverage(10)
        self.position_window = RollingWindow[float](2)
        self.alpha = None
        self.beta = None
        if 'tickers' not in kwargs:
            raise Exception('ShareClassMeanReversionAlphaModel: Missing argument: "tickers"')
        self.tickers = kwargs['tickers']
        self.position_value = None
        self.invested = False
        self.liquidate = 'liquidate'
        self.long_symbol = self.tickers[0]
        self.short_symbol = self.tickers[1]
        self.resolution = kwargs['resolution'] if 'resolution' in kwargs else Resolution.Minute
        self.prediction_interval = Time.Multiply(Extensions.ToTimeSpan(self.resolution), 5)
        self.insight_magnitude = 0.001

    def Update(self, algorithm, data):
        if False:
            return 10
        insights = []
        for security in algorithm.Securities:
            if self.DataEventOccured(data, security.Key):
                return insights
        if self.alpha is None or self.beta is None:
            self.CalculateAlphaBeta(algorithm, data)
            algorithm.Log('Alpha: ' + str(self.alpha))
            algorithm.Log('Beta: ' + str(self.beta))
        if not self.sma.IsReady:
            self.UpdateIndicators(data)
            return insights
        self.UpdateIndicators(data)
        if not self.invested:
            if self.position_value >= self.sma.Current.Value:
                insights.append(Insight(self.long_symbol, self.prediction_interval, InsightType.Price, InsightDirection.Down, self.insight_magnitude, None))
                insights.append(Insight(self.short_symbol, self.prediction_interval, InsightType.Price, InsightDirection.Up, self.insight_magnitude, None))
                self.invested = True
            elif self.position_value < self.sma.Current.Value:
                insights.append(Insight(self.long_symbol, self.prediction_interval, InsightType.Price, InsightDirection.Up, self.insight_magnitude, None))
                insights.append(Insight(self.short_symbol, self.prediction_interval, InsightType.Price, InsightDirection.Down, self.insight_magnitude, None))
                self.invested = True
        elif self.invested and self.CrossedMean():
            self.invested = False
        return Insight.Group(insights)

    def DataEventOccured(self, data, symbol):
        if False:
            print('Hello World!')
        if data.Splits.ContainsKey(symbol) or data.Dividends.ContainsKey(symbol) or data.Delistings.ContainsKey(symbol) or data.SymbolChangedEvents.ContainsKey(symbol):
            return True

    def UpdateIndicators(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.position_value = self.alpha * data[self.long_symbol].Close - self.beta * data[self.short_symbol].Close
        self.sma.Update(data[self.long_symbol].EndTime, self.position_value)
        self.position_window.Add(self.position_value)

    def CrossedMean(self):
        if False:
            print('Hello World!')
        if self.position_window[0] >= self.sma.Current.Value and self.position_window[1] < self.sma.Current.Value:
            return True
        elif self.position_window[0] < self.sma.Current.Value and self.position_window[1] >= self.sma.Current.Value:
            return True
        else:
            return False

    def CalculateAlphaBeta(self, algorithm, data):
        if False:
            i = 10
            return i + 15
        self.alpha = algorithm.CalculateOrderQuantity(self.long_symbol, 0.5)
        self.beta = algorithm.CalculateOrderQuantity(self.short_symbol, 0.5)