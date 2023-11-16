from AlgorithmImports import *

class StandardDeviationExecutionModel(ExecutionModel):
    """Execution model that submits orders while the current market prices is at least the configured number of standard
     deviations away from the mean in the favorable direction (below/above for buy/sell respectively)"""

    def __init__(self, period=60, deviations=2, resolution=Resolution.Minute):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a new instance of the StandardDeviationExecutionModel class\n        Args:\n            period: Period of the standard deviation indicator\n            deviations: The number of deviations away from the mean before submitting an order\n            resolution: The resolution of the STD and SMA indicators'
        self.period = period
        self.deviations = deviations
        self.resolution = resolution
        self.targetsCollection = PortfolioTargetCollection()
        self.symbolData = {}
        self.MaximumOrderValue = 20000

    def Execute(self, algorithm, targets):
        if False:
            while True:
                i = 10
        'Executes market orders if the standard deviation of price is more\n       than the configured number of deviations in the favorable direction.\n       Args:\n           algorithm: The algorithm instance\n           targets: The portfolio targets'
        self.targetsCollection.AddRange(targets)
        if not self.targetsCollection.IsEmpty:
            for target in self.targetsCollection.OrderByMarginImpact(algorithm):
                symbol = target.Symbol
                unorderedQuantity = OrderSizing.GetUnorderedQuantity(algorithm, target)
                data = self.symbolData.get(symbol, None)
                if data is None:
                    return
                if data.STD.IsReady and self.PriceIsFavorable(data, unorderedQuantity):
                    orderSize = OrderSizing.GetOrderSizeForMaximumValue(data.Security, self.MaximumOrderValue, unorderedQuantity)
                    if orderSize != 0:
                        algorithm.MarketOrder(symbol, orderSize)
            self.targetsCollection.ClearFulfilled(algorithm)

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            for i in range(10):
                print('nop')
        'Event fired each time the we add/remove securities from the data feed\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        for added in changes.AddedSecurities:
            if added.Symbol not in self.symbolData:
                self.symbolData[added.Symbol] = SymbolData(algorithm, added, self.period, self.resolution)
        for removed in changes.RemovedSecurities:
            symbol = removed.Symbol
            if symbol in self.symbolData:
                if self.IsSafeToRemove(algorithm, symbol):
                    data = self.symbolData.pop(symbol)
                    algorithm.SubscriptionManager.RemoveConsolidator(symbol, data.Consolidator)

    def PriceIsFavorable(self, data, unorderedQuantity):
        if False:
            while True:
                i = 10
        'Determines if the current price is more than the configured\n       number of standard deviations away from the mean in the favorable direction.'
        sma = data.SMA.Current.Value
        deviations = self.deviations * data.STD.Current.Value
        if unorderedQuantity > 0:
            return data.Security.BidPrice < sma - deviations
        else:
            return data.Security.AskPrice > sma + deviations

    def IsSafeToRemove(self, algorithm, symbol):
        if False:
            print('Hello World!')
        "Determines if it's safe to remove the associated symbol data"
        return not any([kvp.Value.ContainsMember(symbol) for kvp in algorithm.UniverseManager])

class SymbolData:

    def __init__(self, algorithm, security, period, resolution):
        if False:
            return 10
        symbol = security.Symbol
        self.Security = security
        self.Consolidator = algorithm.ResolveConsolidator(symbol, resolution)
        smaName = algorithm.CreateIndicatorName(symbol, f'SMA{period}', resolution)
        self.SMA = SimpleMovingAverage(smaName, period)
        algorithm.RegisterIndicator(symbol, self.SMA, self.Consolidator)
        stdName = algorithm.CreateIndicatorName(symbol, f'STD{period}', resolution)
        self.STD = StandardDeviation(stdName, period)
        algorithm.RegisterIndicator(symbol, self.STD, self.Consolidator)
        bars = algorithm.History[self.Consolidator.InputType](symbol, period, resolution)
        for bar in bars:
            self.SMA.Update(bar.EndTime, bar.Close)
            self.STD.Update(bar.EndTime, bar.Close)