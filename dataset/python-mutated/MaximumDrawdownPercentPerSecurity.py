from AlgorithmImports import *

class MaximumDrawdownPercentPerSecurity(RiskManagementModel):
    """Provides an implementation of IRiskManagementModel that limits the drawdown per holding to the specified percentage"""

    def __init__(self, maximumDrawdownPercent=0.05):
        if False:
            print('Hello World!')
        'Initializes a new instance of the MaximumDrawdownPercentPerSecurity class\n        Args:\n            maximumDrawdownPercent: The maximum percentage drawdown allowed for any single security holding'
        self.maximumDrawdownPercent = -abs(maximumDrawdownPercent)

    def ManageRisk(self, algorithm, targets):
        if False:
            i = 10
            return i + 15
        "Manages the algorithm's risk at each time step\n        Args:\n            algorithm: The algorithm instance\n            targets: The current portfolio targets to be assessed for risk"
        targets = []
        for kvp in algorithm.Securities:
            security = kvp.Value
            if not security.Invested:
                continue
            pnl = security.Holdings.UnrealizedProfitPercent
            if pnl < self.maximumDrawdownPercent:
                symbol = security.Symbol
                algorithm.Insights.Cancel([symbol])
                targets.append(PortfolioTarget(symbol, 0))
        return targets