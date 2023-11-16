from AlgorithmImports import *

class MaximumUnrealizedProfitPercentPerSecurity(RiskManagementModel):
    """Provides an implementation of IRiskManagementModel that limits the unrealized profit per holding to the specified percentage"""

    def __init__(self, maximumUnrealizedProfitPercent=0.05):
        if False:
            return 10
        'Initializes a new instance of the MaximumUnrealizedProfitPercentPerSecurity class\n        Args:\n            maximumUnrealizedProfitPercent: The maximum percentage unrealized profit allowed for any single security holding, defaults to 5% drawdown per security'
        self.maximumUnrealizedProfitPercent = abs(maximumUnrealizedProfitPercent)

    def ManageRisk(self, algorithm, targets):
        if False:
            while True:
                i = 10
        "Manages the algorithm's risk at each time step\n        Args:\n            algorithm: The algorithm instance\n            targets: The current portfolio targets to be assessed for risk"
        targets = []
        for kvp in algorithm.Securities:
            security = kvp.Value
            if not security.Invested:
                continue
            pnl = security.Holdings.UnrealizedProfitPercent
            if pnl > self.maximumUnrealizedProfitPercent:
                symbol = security.Symbol
                algorithm.Insights.Cancel([symbol])
                targets.append(PortfolioTarget(symbol, 0))
        return targets