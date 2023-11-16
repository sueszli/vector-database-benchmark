from AlgorithmImports import *
from Selection.FundamentalUniverseSelectionModel import FundamentalUniverseSelectionModel

class SykesShortMicroCapAlpha(QCAlgorithm):
    """ Alpha Streams: Benchmark Alpha: Identify "pumped" penny stocks and predict that the price of a "pumped" penny stock reverts to mean

    This alpha is part of the Benchmark Alpha Series created by QuantConnect which are open
   sourced so the community and client funds can see an example of an alpha."""

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2018, 1, 1)
        self.SetCash(100000)
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetUniverseSelection(PennyStockUniverseSelectionModel())
        self.SetAlpha(SykesShortMicroCapAlphaModel())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

class SykesShortMicroCapAlphaModel(AlphaModel):
    """Uses ranking of intraday percentage difference between open price and close price to create magnitude and direction prediction for insights"""

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        lookback = kwargs['lookback'] if 'lookback' in kwargs else 1
        resolution = kwargs['resolution'] if 'resolution' in kwargs else Resolution.Daily
        self.predictionInterval = Time.Multiply(Extensions.ToTimeSpan(resolution), lookback)
        self.numberOfStocks = kwargs['numberOfStocks'] if 'numberOfStocks' in kwargs else 10

    def Update(self, algorithm, data):
        if False:
            return 10
        insights = []
        symbolsRet = dict()
        for security in algorithm.ActiveSecurities.Values:
            if security.HasData:
                open = security.Open
                if open != 0:
                    symbolsRet[security.Symbol] = security.Close / open - 1
        pumpedStocks = dict(sorted(symbolsRet.items(), key=lambda kv: (-round(kv[1], 6), kv[0]))[:self.numberOfStocks])
        for (symbol, value) in pumpedStocks.items():
            insights.append(Insight.Price(symbol, self.predictionInterval, InsightDirection.Down, abs(value), None))
        return insights

class PennyStockUniverseSelectionModel(FundamentalUniverseSelectionModel):
    """Defines a universe of penny stocks, as a universe selection model for the framework algorithm:
    The stocks must have fundamental data
    The stock must have positive previous-day close price
    The stock must have volume between $1000000 and $10000 on the previous trading day
    The stock must cost less than $5"""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(False)
        self.numberOfSymbolsCoarse = 500
        self.lastMonth = -1

    def SelectCoarse(self, algorithm, coarse):
        if False:
            return 10
        if algorithm.Time.month == self.lastMonth:
            return Universe.Unchanged
        self.lastMonth = algorithm.Time.month
        top = sorted([x for x in coarse if x.HasFundamentalData and 5 > x.Price > 0 and (1000000 > x.Volume > 10000)], key=lambda x: x.DollarVolume, reverse=True)[:self.numberOfSymbolsCoarse]
        return [x.Symbol for x in top]