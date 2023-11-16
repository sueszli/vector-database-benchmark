from AlgorithmImports import *

class LiquidETFUniverseFrameworkAlgorithm(QCAlgorithm):
    """Basic template framework algorithm uses framework components to define the algorithm."""

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2014, 11, 1)
        self.SetCash(1000000)
        self.SetBenchmark('SPY')
        self.SetBrokerageModel(AlphaStreamsBrokerageModel())
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetUniverseSelection(LiquidETFUniverse())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.symbols = []

    def OnData(self, slice):
        if False:
            while True:
                i = 10
        if all([self.Portfolio[x].Invested for x in self.symbols]):
            return
        insights = [Insight.Price(x, timedelta(1), InsightDirection.Up) for x in self.symbols if self.Securities[x].Price > 0]
        if len(insights) > 0:
            self.EmitInsights(insights)

    def OnSecuritiesChanged(self, changes):
        if False:
            while True:
                i = 10
        for security in changes.AddedSecurities:
            if security.Symbol in LiquidETFUniverse.Energy.Inverse:
                self.symbols.append(security.Symbol)
        self.Log(f'Energy: {LiquidETFUniverse.Energy}')
        self.Log(f'Metals: {LiquidETFUniverse.Metals}')
        self.Log(f'Technology: {LiquidETFUniverse.Technology}')
        self.Log(f'Treasuries: {LiquidETFUniverse.Treasuries}')
        self.Log(f'Volatility: {LiquidETFUniverse.Volatility}')
        self.Log(f'SP500Sectors: {LiquidETFUniverse.SP500Sectors}')