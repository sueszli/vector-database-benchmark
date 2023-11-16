from AlgorithmImports import *

class ImmediateExecutionModel(ExecutionModel):
    """Provides an implementation of IExecutionModel that immediately submits market orders to achieve the desired portfolio targets"""

    def __init__(self):
        if False:
            while True:
                i = 10
        'Initializes a new instance of the ImmediateExecutionModel class'
        self.targetsCollection = PortfolioTargetCollection()

    def Execute(self, algorithm, targets):
        if False:
            print('Hello World!')
        'Immediately submits orders for the specified portfolio targets.\n        Args:\n            algorithm: The algorithm instance\n            targets: The portfolio targets to be ordered'
        self.targetsCollection.AddRange(targets)
        if not self.targetsCollection.IsEmpty:
            for target in self.targetsCollection.OrderByMarginImpact(algorithm):
                security = algorithm.Securities[target.Symbol]
                quantity = OrderSizing.GetUnorderedQuantity(algorithm, target, security)
                if quantity != 0:
                    aboveMinimumPortfolio = BuyingPowerModelExtensions.AboveMinimumOrderMarginPortfolioPercentage(security.BuyingPowerModel, security, quantity, algorithm.Portfolio, algorithm.Settings.MinimumOrderMarginPortfolioPercentage)
                    if aboveMinimumPortfolio:
                        algorithm.MarketOrder(security, quantity)
                    elif not PortfolioTarget.MinimumOrderMarginPercentageWarningSent:
                        PortfolioTarget.MinimumOrderMarginPercentageWarningSent = False
            self.targetsCollection.ClearFulfilled(algorithm)