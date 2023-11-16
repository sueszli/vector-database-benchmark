from AlgorithmImports import *

class VolumeWeightedAveragePriceExecutionModel(ExecutionModel):
    """Execution model that submits orders while the current market price is more favorable that the current volume weighted average price."""

    def __init__(self):
        if False:
            print('Hello World!')
        'Initializes a new instance of the VolumeWeightedAveragePriceExecutionModel class'
        self.targetsCollection = PortfolioTargetCollection()
        self.symbolData = {}
        self.MaximumOrderQuantityPercentVolume = 0.01

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
                if self.PriceIsFavorable(data, unorderedQuantity):
                    orderSize = OrderSizing.GetOrderSizeForPercentVolume(data.Security, self.MaximumOrderQuantityPercentVolume, unorderedQuantity)
                    if orderSize != 0:
                        algorithm.MarketOrder(symbol, orderSize)
            self.targetsCollection.ClearFulfilled(algorithm)

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            return 10
        'Event fired each time the we add/remove securities from the data feed\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        for removed in changes.RemovedSecurities:
            if removed.Symbol in self.symbolData:
                if self.IsSafeToRemove(algorithm, removed.Symbol):
                    data = self.symbolData.pop(removed.Symbol)
                    algorithm.SubscriptionManager.RemoveConsolidator(removed.Symbol, data.Consolidator)
        for added in changes.AddedSecurities:
            if added.Symbol not in self.symbolData:
                self.symbolData[added.Symbol] = SymbolData(algorithm, added)

    def PriceIsFavorable(self, data, unorderedQuantity):
        if False:
            i = 10
            return i + 15
        'Determines if the current price is more than the configured\n       number of standard deviations away from the mean in the favorable direction.'
        if unorderedQuantity > 0:
            if data.Security.BidPrice < data.VWAP:
                return True
        elif data.Security.AskPrice > data.VWAP:
            return True
        return False

    def IsSafeToRemove(self, algorithm, symbol):
        if False:
            i = 10
            return i + 15
        "Determines if it's safe to remove the associated symbol data"
        return not any([kvp.Value.ContainsMember(symbol) for kvp in algorithm.UniverseManager])

class SymbolData:

    def __init__(self, algorithm, security):
        if False:
            while True:
                i = 10
        self.Security = security
        self.Consolidator = algorithm.ResolveConsolidator(security.Symbol, security.Resolution)
        name = algorithm.CreateIndicatorName(security.Symbol, 'VWAP', security.Resolution)
        self.vwap = IntradayVwap(name)
        algorithm.RegisterIndicator(security.Symbol, self.vwap, self.Consolidator)

    @property
    def VWAP(self):
        if False:
            print('Hello World!')
        return self.vwap.Value

class IntradayVwap:
    """Defines the canonical intraday VWAP indicator"""

    def __init__(self, name):
        if False:
            return 10
        self.Name = name
        self.Value = 0.0
        self.lastDate = datetime.min
        self.sumOfVolume = 0.0
        self.sumOfPriceTimesVolume = 0.0

    @property
    def IsReady(self):
        if False:
            while True:
                i = 10
        return self.sumOfVolume > 0.0

    def Update(self, input):
        if False:
            i = 10
            return i + 15
        'Computes the new VWAP'
        (success, volume, averagePrice) = self.GetVolumeAndAveragePrice(input)
        if not success:
            return self.IsReady
        if self.lastDate != input.EndTime.date():
            self.sumOfVolume = 0.0
            self.sumOfPriceTimesVolume = 0.0
            self.lastDate = input.EndTime.date()
        self.sumOfVolume += volume
        self.sumOfPriceTimesVolume += averagePrice * volume
        if self.sumOfVolume == 0.0:
            self.Value = input.Value
            return self.IsReady
        self.Value = self.sumOfPriceTimesVolume / self.sumOfVolume
        return self.IsReady

    def GetVolumeAndAveragePrice(self, input):
        if False:
            return 10
        'Determines the volume and price to be used for the current input in the VWAP computation'
        if type(input) is Tick:
            if input.TickType == TickType.Trade:
                return (True, float(input.Quantity), float(input.LastPrice))
        if type(input) is TradeBar:
            if not input.IsFillForward:
                averagePrice = float(input.High + input.Low + input.Close) / 3
                return (True, float(input.Volume), averagePrice)
        return (False, 0.0, 0.0)