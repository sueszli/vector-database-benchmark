import typing
from AlgorithmImports import *
from datetime import timedelta

class ETFConstituentUniverseRSIAlphaModelAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2020, 12, 1)
        self.SetEndDate(2021, 1, 31)
        self.SetCash(100000)
        self.SetAlpha(ConstituentWeightedRsiAlphaModel())
        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        spy = self.AddEquity('SPY', Resolution.Hour).Symbol
        self.UniverseSettings.Resolution = Resolution.Hour
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0.01
        self.AddUniverse(self.Universe.ETF(spy, self.UniverseSettings, self.FilterETFConstituents))

    def FilterETFConstituents(self, constituents):
        if False:
            print('Hello World!')
        return [i.Symbol for i in constituents if i.Weight is not None and i.Weight >= 0.001]

class ConstituentWeightedRsiAlphaModel(AlphaModel):

    def __init__(self, maxTrades=None):
        if False:
            for i in range(10):
                print('nop')
        self.rsiSymbolData = {}

    def Update(self, algorithm: QCAlgorithm, data: Slice):
        if False:
            for i in range(10):
                print('nop')
        algoConstituents = []
        for barSymbol in data.Bars.Keys:
            if not algorithm.Securities[barSymbol].Cache.HasData(ETFConstituentData):
                continue
            constituentData = algorithm.Securities[barSymbol].Cache.GetData[ETFConstituentData]()
            algoConstituents.append(constituentData)
        if len(algoConstituents) == 0 or len(data.Bars) == 0:
            return []
        constituents = {i.Symbol: i for i in algoConstituents}
        for bar in data.Bars.Values:
            if bar.Symbol not in constituents:
                continue
            if bar.Symbol not in self.rsiSymbolData:
                constituent = constituents[bar.Symbol]
                self.rsiSymbolData[bar.Symbol] = SymbolData(bar.Symbol, algorithm, constituent, 7)
        allReady = all([sd.rsi.IsReady for sd in self.rsiSymbolData.values()])
        if not allReady:
            return []
        insights = []
        for (symbol, symbolData) in self.rsiSymbolData.items():
            averageLoss = symbolData.rsi.AverageLoss.Current.Value
            averageGain = symbolData.rsi.AverageGain.Current.Value
            direction = InsightDirection.Down if averageLoss > averageGain else InsightDirection.Up
            insights.append(Insight.Price(symbol, timedelta(days=1), direction, float(averageLoss if direction == InsightDirection.Down else averageGain), weight=float(symbolData.constituent.Weight)))
        return insights

class SymbolData:

    def __init__(self, symbol, algorithm, constituent, period):
        if False:
            while True:
                i = 10
        self.Symbol = symbol
        self.constituent = constituent
        self.rsi = algorithm.RSI(symbol, period, MovingAverageType.Exponential, Resolution.Hour)