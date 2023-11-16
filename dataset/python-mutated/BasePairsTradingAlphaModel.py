from AlgorithmImports import *
from enum import Enum

class BasePairsTradingAlphaModel(AlphaModel):
    """This alpha model is designed to accept every possible pair combination
    from securities selected by the universe selection model
    This model generates alternating long ratio/short ratio insights emitted as a group"""

    def __init__(self, lookback=1, resolution=Resolution.Daily, threshold=1):
        if False:
            print('Hello World!')
        ' Initializes a new instance of the PairsTradingAlphaModel class\n        Args:\n            lookback: Lookback period of the analysis\n            resolution: Analysis resolution\n            threshold: The percent [0, 100] deviation of the ratio from the mean before emitting an insight'
        self.lookback = lookback
        self.resolution = resolution
        self.threshold = threshold
        self.predictionInterval = Time.Multiply(Extensions.ToTimeSpan(self.resolution), self.lookback)
        self.pairs = dict()
        self.Securities = list()
        resolutionString = Extensions.GetEnumString(resolution, Resolution)
        self.Name = f'{self.__class__.__name__}({self.lookback},{resolutionString},{Extensions.NormalizeToStr(threshold)})'

    def Update(self, algorithm, data):
        if False:
            while True:
                i = 10
        ' Updates this alpha model with the latest data from the algorithm.\n        This is called each time the algorithm receives data for subscribed securities\n        Args:\n            algorithm: The algorithm instance\n            data: The new data available\n        Returns:\n            The new insights generated'
        insights = []
        for (key, pair) in self.pairs.items():
            insights.extend(pair.GetInsightGroup())
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            for i in range(10):
                print('nop')
        'Event fired each time the we add/remove securities from the data feed.\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        for security in changes.AddedSecurities:
            self.Securities.append(security)
        for security in changes.RemovedSecurities:
            if security in self.Securities:
                self.Securities.remove(security)
        self.UpdatePairs(algorithm)
        for security in changes.RemovedSecurities:
            keys = [k for k in self.pairs.keys() if security.Symbol in k]
            for key in keys:
                self.pairs.pop(key).dispose()

    def UpdatePairs(self, algorithm):
        if False:
            while True:
                i = 10
        symbols = sorted([x.Symbol for x in self.Securities], key=lambda x: str(x.ID))
        for i in range(0, len(symbols)):
            asset_i = symbols[i]
            for j in range(1 + i, len(symbols)):
                asset_j = symbols[j]
                pair_symbol = (asset_i, asset_j)
                invert = (asset_j, asset_i)
                if pair_symbol in self.pairs or invert in self.pairs:
                    continue
                if not self.HasPassedTest(algorithm, asset_i, asset_j):
                    continue
                pair = self.Pair(algorithm, asset_i, asset_j, self.predictionInterval, self.threshold)
                self.pairs[pair_symbol] = pair

    def HasPassedTest(self, algorithm, asset1, asset2):
        if False:
            return 10
        "Check whether the assets pass a pairs trading test\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            asset1: The first asset's symbol in the pair\n            asset2: The second asset's symbol in the pair\n        Returns:\n            True if the statistical test for the pair is successful"
        return True

    class Pair:

        class State(Enum):
            ShortRatio = -1
            FlatRatio = 0
            LongRatio = 1

        def __init__(self, algorithm, asset1, asset2, predictionInterval, threshold):
            if False:
                while True:
                    i = 10
            "Create a new pair\n            Args:\n                algorithm: The algorithm instance that experienced the change in securities\n                asset1: The first asset's symbol in the pair\n                asset2: The second asset's symbol in the pair\n                predictionInterval: Period over which this insight is expected to come to fruition\n                threshold: The percent [0, 100] deviation of the ratio from the mean before emitting an insight"
            self.state = self.State.FlatRatio
            self.algorithm = algorithm
            self.asset1 = asset1
            self.asset2 = asset2

            def CreateIdentityIndicator(symbol: Symbol):
                if False:
                    while True:
                        i = 10
                resolution = min([x.Resolution for x in algorithm.SubscriptionManager.SubscriptionDataConfigService.GetSubscriptionDataConfigs(symbol)])
                name = algorithm.CreateIndicatorName(symbol, 'close', resolution)
                identity = Identity(name)
                consolidator = algorithm.ResolveConsolidator(symbol, resolution)
                algorithm.RegisterIndicator(symbol, identity, consolidator)
                return (identity, consolidator)
            (self.asset1Price, self.identityConsolidator1) = CreateIdentityIndicator(asset1)
            (self.asset2Price, self.identityConsolidator2) = CreateIdentityIndicator(asset2)
            self.ratio = IndicatorExtensions.Over(self.asset1Price, self.asset2Price)
            self.mean = IndicatorExtensions.Of(ExponentialMovingAverage(500), self.ratio)
            upper = ConstantIndicator[IndicatorDataPoint]('ct', 1 + threshold / 100)
            self.upperThreshold = IndicatorExtensions.Times(self.mean, upper)
            lower = ConstantIndicator[IndicatorDataPoint]('ct', 1 - threshold / 100)
            self.lowerThreshold = IndicatorExtensions.Times(self.mean, lower)
            self.predictionInterval = predictionInterval

        def dispose(self):
            if False:
                return 10
            '\n            On disposal, remove the consolidators from the subscription manager\n            '
            self.algorithm.SubscriptionManager.RemoveConsolidator(self.asset1, self.identityConsolidator1)
            self.algorithm.SubscriptionManager.RemoveConsolidator(self.asset2, self.identityConsolidator2)

        def GetInsightGroup(self):
            if False:
                return 10
            'Gets the insights group for the pair\n            Returns:\n                Insights grouped by an unique group id'
            if not self.mean.IsReady:
                return []
            if self.state is not self.State.LongRatio and self.ratio > self.upperThreshold:
                self.state = self.State.LongRatio
                shortAsset1 = Insight.Price(self.asset1, self.predictionInterval, InsightDirection.Down)
                longAsset2 = Insight.Price(self.asset2, self.predictionInterval, InsightDirection.Up)
                return Insight.Group(shortAsset1, longAsset2)
            if self.state is not self.State.ShortRatio and self.ratio < self.lowerThreshold:
                self.state = self.State.ShortRatio
                longAsset1 = Insight.Price(self.asset1, self.predictionInterval, InsightDirection.Up)
                shortAsset2 = Insight.Price(self.asset2, self.predictionInterval, InsightDirection.Down)
                return Insight.Group(longAsset1, shortAsset2)
            return []