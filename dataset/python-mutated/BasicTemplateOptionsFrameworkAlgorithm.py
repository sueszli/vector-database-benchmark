from AlgorithmImports import *
from Alphas.ConstantAlphaModel import ConstantAlphaModel
from Selection.OptionUniverseSelectionModel import OptionUniverseSelectionModel
from Execution.ImmediateExecutionModel import ImmediateExecutionModel
from Risk.NullRiskManagementModel import NullRiskManagementModel

class BasicTemplateOptionsFrameworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetStartDate(2014, 6, 5)
        self.SetEndDate(2014, 6, 9)
        self.SetCash(100000)
        self.SetUniverseSelection(EarliestExpiringWeeklyAtTheMoneyPutOptionUniverseSelectionModel(self.SelectOptionChainSymbols))
        self.SetAlpha(ConstantOptionContractAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(hours=0.5)))
        self.SetPortfolioConstruction(SingleSharePortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

    def SelectOptionChainSymbols(self, utcTime):
        if False:
            print('Hello World!')
        newYorkTime = Extensions.ConvertFromUtc(utcTime, TimeZones.NewYork)
        ticker = 'TWX' if newYorkTime.date() < date(2014, 6, 6) else 'AAPL'
        return [Symbol.Create(ticker, SecurityType.Option, Market.USA, f'?{ticker}')]

class EarliestExpiringWeeklyAtTheMoneyPutOptionUniverseSelectionModel(OptionUniverseSelectionModel):
    """Creates option chain universes that select only the earliest expiry ATM weekly put contract
    and runs a user defined optionChainSymbolSelector every day to enable choosing different option chains"""

    def __init__(self, select_option_chain_symbols):
        if False:
            i = 10
            return i + 15
        super().__init__(timedelta(1), select_option_chain_symbols)

    def Filter(self, filter):
        if False:
            print('Hello World!')
        'Defines the option chain universe filter'
        return filter.Strikes(+1, +1).Expiration(0, 7).WeeklysOnly().PutsOnly().OnlyApplyFilterAtMarketOpen()

class ConstantOptionContractAlphaModel(ConstantAlphaModel):
    """Implementation of a constant alpha model that only emits insights for option symbols"""

    def __init__(self, type, direction, period):
        if False:
            while True:
                i = 10
        super().__init__(type, direction, period)

    def ShouldEmitInsight(self, utcTime, symbol):
        if False:
            for i in range(10):
                print('nop')
        if symbol.SecurityType != SecurityType.Option:
            return False
        return super().ShouldEmitInsight(utcTime, symbol)

class SingleSharePortfolioConstructionModel(PortfolioConstructionModel):
    """Portfolio construction model that sets target quantities to 1 for up insights and -1 for down insights"""

    def CreateTargets(self, algorithm, insights):
        if False:
            return 10
        targets = []
        for insight in insights:
            targets.append(PortfolioTarget(insight.Symbol, insight.Direction))
        return targets