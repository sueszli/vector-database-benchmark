from AlgorithmImports import *

class LongOnlyAlphaStreamAlgorithm(QCAlgorithm):
    """Basic template framework algorithm uses framework components to define the algorithm.
    Shows EqualWeightingPortfolioConstructionModel.LongOnly() application"""

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetBrokerageModel(BrokerageName.AlphaStreams)
        self.SetCash(1000000)
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel(Resolution.Daily, PortfolioBias.Long))
        self.SetExecution(ImmediateExecutionModel())
        self.SetUniverseSelection(ManualUniverseSelectionModel([Symbol.Create(x, SecurityType.Equity, Market.USA) for x in ['SPY', 'IBM']]))

    def OnData(self, slice):
        if False:
            for i in range(10):
                print('nop')
        if self.Portfolio.Invested:
            return
        self.EmitInsights([Insight.Price('SPY', timedelta(1), InsightDirection.Up), Insight.Price('IBM', timedelta(1), InsightDirection.Down)])

    def OnOrderEvent(self, orderEvent):
        if False:
            i = 10
            return i + 15
        if orderEvent.Status == OrderStatus.Filled:
            if self.Securities[orderEvent.Symbol].Holdings.IsShort:
                raise ValueError('Invalid position, should not be short')
            self.Debug(orderEvent)