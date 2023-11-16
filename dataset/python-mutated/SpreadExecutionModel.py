from AlgorithmImports import *

class SpreadExecutionModel(ExecutionModel):
    """Execution model that submits orders while the current spread is tight.
       Note this execution model will not work using Resolution.Daily since Exchange.ExchangeOpen will be false, suggested resolution is Minute
    """

    def __init__(self, acceptingSpreadPercent=0.005):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a new instance of the SpreadExecutionModel class'
        self.targetsCollection = PortfolioTargetCollection()
        self.acceptingSpreadPercent = Math.Abs(acceptingSpreadPercent)

    def Execute(self, algorithm, targets):
        if False:
            i = 10
            return i + 15
        'Executes market orders if the spread percentage to price is in desirable range.\n       Args:\n           algorithm: The algorithm instance\n           targets: The portfolio targets'
        self.targetsCollection.AddRange(targets)
        if not self.targetsCollection.IsEmpty:
            for target in self.targetsCollection.OrderByMarginImpact(algorithm):
                symbol = target.Symbol
                unorderedQuantity = OrderSizing.GetUnorderedQuantity(algorithm, target)
                if unorderedQuantity != 0:
                    security = algorithm.Securities[symbol]
                    if self.SpreadIsFavorable(security):
                        algorithm.MarketOrder(symbol, unorderedQuantity)
            self.targetsCollection.ClearFulfilled(algorithm)

    def SpreadIsFavorable(self, security):
        if False:
            i = 10
            return i + 15
        'Determines if the spread is in desirable range.'
        return security.Exchange.ExchangeOpen and security.Price > 0 and (security.AskPrice > 0) and (security.BidPrice > 0) and ((security.AskPrice - security.BidPrice) / security.Price <= self.acceptingSpreadPercent)