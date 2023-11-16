from AlgorithmImports import *

class MaximumDrawdownPercentPortfolio(RiskManagementModel):
    """Provides an implementation of IRiskManagementModel that limits the drawdown of the portfolio to the specified percentage."""

    def __init__(self, maximumDrawdownPercent=0.05, isTrailing=False):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a new instance of the MaximumDrawdownPercentPortfolio class\n        Args:\n            maximumDrawdownPercent: The maximum percentage drawdown allowed for algorithm portfolio compared with starting value, defaults to 5% drawdown</param>\n            isTrailing: If "false", the drawdown will be relative to the starting value of the portfolio.\n                        If "true", the drawdown will be relative the last maximum portfolio value'
        self.maximumDrawdownPercent = -abs(maximumDrawdownPercent)
        self.isTrailing = isTrailing
        self.initialised = False
        self.portfolioHigh = 0

    def ManageRisk(self, algorithm, targets):
        if False:
            i = 10
            return i + 15
        "Manages the algorithm's risk at each time step\n        Args:\n            algorithm: The algorithm instance\n            targets: The current portfolio targets to be assessed for risk"
        currentValue = algorithm.Portfolio.TotalPortfolioValue
        if not self.initialised:
            self.portfolioHigh = currentValue
            self.initialised = True
        if self.isTrailing and self.portfolioHigh < currentValue:
            self.portfolioHigh = currentValue
            return []
        pnl = self.GetTotalDrawdownPercent(currentValue)
        if pnl < self.maximumDrawdownPercent and len(targets) != 0:
            self.initialised = False
            risk_adjusted_targets = []
            for target in targets:
                symbol = target.Symbol
                algorithm.Insights.Cancel([symbol])
                risk_adjusted_targets.append(PortfolioTarget(symbol, 0))
            return risk_adjusted_targets
        return []

    def GetTotalDrawdownPercent(self, currentValue):
        if False:
            print('Hello World!')
        return float(currentValue) / float(self.portfolioHigh) - 1.0