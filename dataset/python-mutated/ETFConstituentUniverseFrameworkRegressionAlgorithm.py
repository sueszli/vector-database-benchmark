import typing
from AlgorithmImports import *
constituentData = []

class ETFConstituentAlphaModel(AlphaModel):

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            for i in range(10):
                print('nop')
        pass

    def Update(self, algorithm: QCAlgorithm, data: Slice):
        if False:
            while True:
                i = 10
        insights = []
        for constituent in constituentData:
            if constituent.Symbol not in data.Bars and constituent.Symbol not in data.QuoteBars:
                continue
            insightDirection = InsightDirection.Up if constituent.Weight is not None and constituent.Weight >= 0.01 else InsightDirection.Down
            insights.append(Insight(algorithm.UtcTime, constituent.Symbol, timedelta(days=1), InsightType.Price, insightDirection, float(1 * int(insightDirection)), 1.0, weight=float(0 if constituent.Weight is None else constituent.Weight)))
        return insights

class ETFConstituentPortfolioModel(PortfolioConstructionModel):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.hasAdded = False

    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges):
        if False:
            while True:
                i = 10
        self.hasAdded = len(changes.AddedSecurities) != 0

    def CreateTargets(self, algorithm: QCAlgorithm, insights: typing.List[Insight]):
        if False:
            return 10
        if not self.hasAdded:
            return []
        finalInsights = []
        for insight in insights:
            finalInsights.append(PortfolioTarget(insight.Symbol, float(0 if insight.Weight is None else insight.Weight)))
            self.hasAdded = False
        return finalInsights

class ETFConstituentExecutionModel(ExecutionModel):

    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges):
        if False:
            print('Hello World!')
        for change in changes.RemovedSecurities:
            algorithm.Liquidate(change.Symbol)

    def Execute(self, algorithm: QCAlgorithm, targets: typing.List[IPortfolioTarget]):
        if False:
            while True:
                i = 10
        for target in targets:
            algorithm.SetHoldings(target.Symbol, target.Quantity)

class ETFConstituentUniverseFrameworkRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2020, 12, 1)
        self.SetEndDate(2021, 1, 31)
        self.SetCash(100000)
        self.SetAlpha(ETFConstituentAlphaModel())
        self.SetPortfolioConstruction(ETFConstituentPortfolioModel())
        self.SetExecution(ETFConstituentExecutionModel())
        spy = Symbol.Create('SPY', SecurityType.Equity, Market.USA)
        self.UniverseSettings.Resolution = Resolution.Hour
        self.AddUniverse(self.Universe.ETF(spy, self.UniverseSettings, self.FilterETFConstituents))

    def FilterETFConstituents(self, constituents):
        if False:
            for i in range(10):
                print('nop')
        global constituentData
        constituentDataLocal = [i for i in constituents if i is not None and i.Weight >= 0.001]
        constituentData = list(constituentDataLocal)
        return [i.Symbol for i in constituentDataLocal]

    def OnData(self, data):
        if False:
            while True:
                i = 10
        pass