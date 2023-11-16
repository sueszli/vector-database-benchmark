from AlgorithmImports import *

class PortfolioRebalanceOnDateRulesRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        ' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.UniverseSettings.Resolution = Resolution.Daily
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2017, 1, 1)
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.Settings.RebalancePortfolioOnSecurityChanges = False
        self.SetUniverseSelection(CustomUniverseSelectionModel('CustomUniverseSelectionModel', lambda time: ['AAPL', 'IBM', 'FB', 'SPY']))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, TimeSpan.FromMinutes(20), 0.025, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel(self.DateRules.Every(DayOfWeek.Wednesday)))
        self.SetExecution(ImmediateExecutionModel())

    def OnOrderEvent(self, orderEvent):
        if False:
            while True:
                i = 10
        if orderEvent.Status == OrderStatus.Submitted:
            self.Debug(str(orderEvent))
            if self.UtcTime.weekday() != 2:
                raise ValueError(str(self.UtcTime) + ' ' + str(orderEvent.Symbol) + ' ' + str(self.UtcTime.weekday()))