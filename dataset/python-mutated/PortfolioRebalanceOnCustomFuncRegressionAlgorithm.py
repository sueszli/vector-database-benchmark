from AlgorithmImports import *

class PortfolioRebalanceOnCustomFuncRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        ' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2018, 1, 1)
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.Settings.RebalancePortfolioOnSecurityChanges = False
        self.SetUniverseSelection(CustomUniverseSelectionModel('CustomUniverseSelectionModel', lambda time: ['AAPL', 'IBM', 'FB', 'SPY', 'AIG', 'BAC', 'BNO']))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, TimeSpan.FromMinutes(20), 0.025, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel(self.RebalanceFunction))
        self.SetExecution(ImmediateExecutionModel())
        self.lastRebalanceTime = self.StartDate

    def RebalanceFunction(self, time):
        if False:
            print('Hello World!')
        if time.weekday() != 0:
            return None
        if self.lastRebalanceTime == self.StartDate:
            self.lastRebalanceTime = time
            return time
        deviation = 0
        count = sum((1 for security in self.Securities.Values if security.Invested))
        if count > 0:
            self.lastRebalanceTime = time
            portfolioValuePerSecurity = self.Portfolio.TotalPortfolioValue / count
            for security in self.Securities.Values:
                if not security.Invested:
                    continue
                reservedBuyingPowerForCurrentPosition = security.BuyingPowerModel.GetReservedBuyingPowerForPosition(ReservedBuyingPowerForPositionParameters(security)).AbsoluteUsedBuyingPower * security.BuyingPowerModel.GetLeverage(security)
                deviation += (portfolioValuePerSecurity - reservedBuyingPowerForCurrentPosition) / portfolioValuePerSecurity
            if deviation >= 0.015:
                return time
        return None

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        if orderEvent.Status == OrderStatus.Submitted:
            if self.UtcTime != self.lastRebalanceTime or self.UtcTime.weekday() != 0:
                raise ValueError(f'{self.UtcTime} {orderEvent.Symbol}')