from AlgorithmImports import *

class TriangleExchangeRateArbitrageAlpha(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2019, 2, 1)
        self.SetCash(100000)
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))
        currencies = ['EURUSD', 'EURGBP', 'GBPUSD']
        symbols = [Symbol.Create(currency, SecurityType.Forex, Market.Oanda) for currency in currencies]
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.SetAlpha(ForexTriangleArbitrageAlphaModel(Resolution.Minute, symbols))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

class ForexTriangleArbitrageAlphaModel(AlphaModel):

    def __init__(self, insight_resolution, symbols):
        if False:
            print('Hello World!')
        self.insight_period = Time.Multiply(Extensions.ToTimeSpan(insight_resolution), 5)
        self.symbols = symbols

    def Update(self, algorithm, data):
        if False:
            for i in range(10):
                print('nop')
        if len(data.Keys) < 3:
            return []
        bar_a = data[self.symbols[0]]
        bar_b = data[self.symbols[1]]
        bar_c = data[self.symbols[2]]
        triangleRate = bar_a.Ask.Close / bar_b.Bid.Close / bar_c.Ask.Close
        if triangleRate > 1.0005:
            return Insight.Group([Insight.Price(self.symbols[0], self.insight_period, InsightDirection.Up, 0.0001, None), Insight.Price(self.symbols[1], self.insight_period, InsightDirection.Down, 0.0001, None), Insight.Price(self.symbols[2], self.insight_period, InsightDirection.Up, 0.0001, None)])
        return []