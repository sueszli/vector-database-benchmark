from AlgorithmImports import *

class TrailingStopRiskManagementModel(RiskManagementModel):
    """Provides an implementation of IRiskManagementModel that limits the maximum possible loss
    measured from the highest unrealized profit"""

    def __init__(self, maximumDrawdownPercent=0.05):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a new instance of the TrailingStopRiskManagementModel class\n        Args:\n            maximumDrawdownPercent: The maximum percentage drawdown allowed for algorithm portfolio compared with the highest unrealized profit, defaults to 5% drawdown'
        self.maximumDrawdownPercent = abs(maximumDrawdownPercent)
        self.trailingAbsoluteHoldingsState = dict()

    def ManageRisk(self, algorithm, targets):
        if False:
            for i in range(10):
                print('nop')
        "Manages the algorithm's risk at each time step\n        Args:\n            algorithm: The algorithm instance\n            targets: The current portfolio targets to be assessed for risk"
        riskAdjustedTargets = list()
        for kvp in algorithm.Securities:
            symbol = kvp.Key
            security = kvp.Value
            if not security.Invested:
                self.trailingAbsoluteHoldingsState.pop(symbol, None)
                continue
            position = PositionSide.Long if security.Holdings.IsLong else PositionSide.Short
            absoluteHoldingsValue = security.Holdings.AbsoluteHoldingsValue
            trailingAbsoluteHoldingsState = self.trailingAbsoluteHoldingsState.get(symbol)
            if trailingAbsoluteHoldingsState == None or position != trailingAbsoluteHoldingsState.position:
                self.trailingAbsoluteHoldingsState[symbol] = trailingAbsoluteHoldingsState = self.HoldingsState(position, security.Holdings.AbsoluteHoldingsCost)
            trailingAbsoluteHoldingsValue = trailingAbsoluteHoldingsState.absoluteHoldingsValue
            if position == PositionSide.Long and trailingAbsoluteHoldingsValue < absoluteHoldingsValue or (position == PositionSide.Short and trailingAbsoluteHoldingsValue > absoluteHoldingsValue):
                self.trailingAbsoluteHoldingsState[symbol].absoluteHoldingsValue = absoluteHoldingsValue
                continue
            drawdown = abs((trailingAbsoluteHoldingsValue - absoluteHoldingsValue) / trailingAbsoluteHoldingsValue)
            if self.maximumDrawdownPercent < drawdown:
                algorithm.Insights.Cancel([symbol])
                self.trailingAbsoluteHoldingsState.pop(symbol, None)
                riskAdjustedTargets.append(PortfolioTarget(symbol, 0))
        return riskAdjustedTargets

    class HoldingsState:

        def __init__(self, position, absoluteHoldingsValue):
            if False:
                while True:
                    i = 10
            self.position = position
            self.absoluteHoldingsValue = absoluteHoldingsValue