from AlgorithmImports import *
from Alphas.ConstantAlphaModel import ConstantAlphaModel
from Selection.FutureUniverseSelectionModel import FutureUniverseSelectionModel

class BasicTemplateFuturesFrameworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.UniverseSettings.Resolution = Resolution.Minute
        self.UniverseSettings.ExtendedMarketHours = self.GetExtendedMarketHours()
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.SetUniverseSelection(FrontMonthFutureUniverseSelectionModel(self.SelectFutureChainSymbols))
        self.SetAlpha(ConstantFutureContractAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(1)))
        self.SetPortfolioConstruction(SingleSharePortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

    def SelectFutureChainSymbols(self, utcTime):
        if False:
            return 10
        newYorkTime = Extensions.ConvertFromUtc(utcTime, TimeZones.NewYork)
        if newYorkTime.date() < date(2013, 10, 9):
            return [Symbol.Create(Futures.Indices.SP500EMini, SecurityType.Future, Market.CME)]
        else:
            return [Symbol.Create(Futures.Metals.Gold, SecurityType.Future, Market.COMEX)]

    def GetExtendedMarketHours(self):
        if False:
            return 10
        return False

class FrontMonthFutureUniverseSelectionModel(FutureUniverseSelectionModel):
    """Creates futures chain universes that select the front month contract and runs a user
    defined futureChainSymbolSelector every day to enable choosing different futures chains"""

    def __init__(self, select_future_chain_symbols):
        if False:
            while True:
                i = 10
        super().__init__(timedelta(1), select_future_chain_symbols)

    def Filter(self, filter):
        if False:
            i = 10
            return i + 15
        'Defines the futures chain universe filter'
        return filter.FrontMonth().OnlyApplyFilterAtMarketOpen()

class ConstantFutureContractAlphaModel(ConstantAlphaModel):
    """Implementation of a constant alpha model that only emits insights for future symbols"""

    def __init__(self, type, direction, period):
        if False:
            print('Hello World!')
        super().__init__(type, direction, period)

    def ShouldEmitInsight(self, utcTime, symbol):
        if False:
            for i in range(10):
                print('nop')
        if symbol.SecurityType != SecurityType.Future:
            return False
        return super().ShouldEmitInsight(utcTime, symbol)

class SingleSharePortfolioConstructionModel(PortfolioConstructionModel):
    """Portfolio construction model that sets target quantities to 1 for up insights and -1 for down insights"""

    def CreateTargets(self, algorithm, insights):
        if False:
            i = 10
            return i + 15
        targets = []
        for insight in insights:
            targets.append(PortfolioTarget(insight.Symbol, insight.Direction))
        return targets