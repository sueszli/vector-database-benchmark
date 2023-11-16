from AlgorithmImports import *

class BasicTemplateFrameworkAlgorithm(QCAlgorithm):
    """Basic template framework algorithm uses framework components to define the algorithm."""

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        ' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        symbols = [Symbol.Create('SPY', SecurityType.Equity, Market.USA)]
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(minutes=20), 0.025, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel(Resolution.Daily))
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.01))
        self.Debug('numpy test >>> print numpy.pi: ' + str(np.pi))

    def OnOrderEvent(self, orderEvent):
        if False:
            while True:
                i = 10
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug('Purchased Stock: {0}'.format(orderEvent.Symbol))