from AlgorithmImports import *

class GlobalEquityMeanReversionIBSAlpha(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2018, 1, 1)
        self.SetCash(100000)
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))
        tickers = ['ECH', 'EEM', 'EFA', 'EPHE', 'EPP', 'EWA', 'EWC', 'EWG', 'EWH', 'EWI', 'EWJ', 'EWL', 'EWM', 'EWM', 'EWO', 'EWP', 'EWQ', 'EWS', 'EWT', 'EWU', 'EWY', 'EWZ', 'EZA', 'FXI', 'GXG', 'IDX', 'ILF', 'EWM', 'QQQ', 'RSX', 'SPY', 'THD']
        symbols = [Symbol.Create(ticker, SecurityType.Equity, Market.USA) for ticker in tickers]
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.SetAlpha(MeanReversionIBSAlphaModel())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

class MeanReversionIBSAlphaModel(AlphaModel):
    """Uses ranking of Internal Bar Strength (IBS) to create direction prediction for insights"""

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        lookback = kwargs['lookback'] if 'lookback' in kwargs else 1
        resolution = kwargs['resolution'] if 'resolution' in kwargs else Resolution.Daily
        self.predictionInterval = Time.Multiply(Extensions.ToTimeSpan(resolution), lookback)
        self.numberOfStocks = kwargs['numberOfStocks'] if 'numberOfStocks' in kwargs else 2

    def Update(self, algorithm, data):
        if False:
            i = 10
            return i + 15
        insights = []
        symbolsIBS = dict()
        returns = dict()
        for security in algorithm.ActiveSecurities.Values:
            if security.HasData:
                high = security.High
                low = security.Low
                hilo = high - low
                if security.Open * hilo != 0:
                    symbolsIBS[security.Symbol] = (security.Close - low) / hilo
                    returns[security.Symbol] = security.Close / security.Open - 1
        number_of_stocks = min(int(len(symbolsIBS) / 2), self.numberOfStocks)
        if number_of_stocks == 0:
            return []
        ordered = sorted(symbolsIBS.items(), key=lambda kv: (round(kv[1], 6), kv[0]), reverse=True)
        highIBS = dict(ordered[0:number_of_stocks])
        lowIBS = dict(ordered[-number_of_stocks:])
        for (key, value) in highIBS.items():
            insights.append(Insight.Price(key, self.predictionInterval, InsightDirection.Down, abs(returns[key]), None))
        for (key, value) in lowIBS.items():
            insights.append(Insight.Price(key, self.predictionInterval, InsightDirection.Up, abs(returns[key]), None))
        return insights