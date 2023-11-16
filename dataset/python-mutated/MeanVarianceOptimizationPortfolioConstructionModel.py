from AlgorithmImports import *
from Portfolio.MinimumVariancePortfolioOptimizer import MinimumVariancePortfolioOptimizer

class MeanVarianceOptimizationPortfolioConstructionModel(PortfolioConstructionModel):

    def __init__(self, rebalance=Resolution.Daily, portfolioBias=PortfolioBias.LongShort, lookback=1, period=63, resolution=Resolution.Daily, targetReturn=0.02, optimizer=None):
        if False:
            print('Hello World!')
        'Initialize the model\n        Args:\n            rebalance: Rebalancing parameter. If it is a timedelta, date rules or Resolution, it will be converted into a function.\n                              If None will be ignored.\n                              The function returns the next expected rebalance time for a given algorithm UTC DateTime.\n                              The function returns null if unknown, in which case the function will be called again in the\n                              next loop. Returning current time will trigger rebalance.\n            portfolioBias: Specifies the bias of the portfolio (Short, Long/Short, Long)\n            lookback(int): Historical return lookback period\n            period(int): The time interval of history price to calculate the weight\n            resolution: The resolution of the history price\n            optimizer(class): Method used to compute the portfolio weights'
        super().__init__()
        self.lookback = lookback
        self.period = period
        self.resolution = resolution
        self.portfolioBias = portfolioBias
        self.sign = lambda x: -1 if x < 0 else 1 if x > 0 else 0
        lower = 0 if portfolioBias == PortfolioBias.Long else -1
        upper = 0 if portfolioBias == PortfolioBias.Short else 1
        self.optimizer = MinimumVariancePortfolioOptimizer(lower, upper, targetReturn) if optimizer is None else optimizer
        self.symbolDataBySymbol = {}
        rebalancingFunc = rebalance
        if isinstance(rebalance, int):
            rebalance = Extensions.ToTimeSpan(rebalance)
        if isinstance(rebalance, timedelta):
            rebalancingFunc = lambda dt: dt + rebalance
        if rebalancingFunc:
            self.SetRebalancingFunc(rebalancingFunc)

    def ShouldCreateTargetForInsight(self, insight):
        if False:
            for i in range(10):
                print('nop')
        if len(PortfolioConstructionModel.FilterInvalidInsightMagnitude(self.Algorithm, [insight])) == 0:
            return False
        symbolData = self.symbolDataBySymbol.get(insight.Symbol)
        if insight.Magnitude is None:
            self.Algorithm.SetRunTimeError(ArgumentNullException("MeanVarianceOptimizationPortfolioConstructionModel does not accept 'None' as Insight.Magnitude. Please checkout the selected Alpha Model specifications."))
            return False
        symbolData.Add(self.Algorithm.Time, insight.Magnitude)
        return True

    def DetermineTargetPercent(self, activeInsights):
        if False:
            while True:
                i = 10
        '\n         Will determine the target percent for each insight\n        Args:\n        Returns:\n        '
        targets = {}
        if len(activeInsights) == 0:
            return targets
        symbols = [insight.Symbol for insight in activeInsights]
        returns = {str(symbol.ID): data.Return for (symbol, data) in self.symbolDataBySymbol.items() if symbol in symbols}
        returns = pd.DataFrame(returns)
        weights = self.optimizer.Optimize(returns)
        weights = pd.Series(weights, index=returns.columns)
        for insight in activeInsights:
            weight = weights[str(insight.Symbol.ID)]
            if self.portfolioBias != PortfolioBias.LongShort and self.sign(weight) != self.portfolioBias:
                weight = 0
            targets[insight] = weight
        return targets

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            i = 10
            return i + 15
        'Event fired each time the we add/remove securities from the data feed\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        super().OnSecuritiesChanged(algorithm, changes)
        for removed in changes.RemovedSecurities:
            symbolData = self.symbolDataBySymbol.pop(removed.Symbol, None)
            symbolData.Reset()
        symbols = [x.Symbol for x in changes.AddedSecurities]
        for symbol in [x for x in symbols if x not in self.symbolDataBySymbol]:
            self.symbolDataBySymbol[symbol] = self.MeanVarianceSymbolData(symbol, self.lookback, self.period)
        history = algorithm.History[TradeBar](symbols, self.lookback * self.period, self.resolution)
        for bars in history:
            for (symbol, bar) in bars.items():
                symbolData = self.symbolDataBySymbol.get(symbol).Update(bar.EndTime, bar.Value)

    class MeanVarianceSymbolData:
        """Contains data specific to a symbol required by this model"""

        def __init__(self, symbol, lookback, period):
            if False:
                print('Hello World!')
            self.symbol = symbol
            self.roc = RateOfChange(f'{symbol}.ROC({lookback})', lookback)
            self.roc.Updated += self.OnRateOfChangeUpdated
            self.window = RollingWindow[IndicatorDataPoint](period)

        def Reset(self):
            if False:
                while True:
                    i = 10
            self.roc.Updated -= self.OnRateOfChangeUpdated
            self.roc.Reset()
            self.window.Reset()

        def Update(self, time, value):
            if False:
                return 10
            return self.roc.Update(time, value)

        def OnRateOfChangeUpdated(self, roc, value):
            if False:
                i = 10
                return i + 15
            if roc.IsReady:
                self.window.Add(value)

        def Add(self, time, value):
            if False:
                i = 10
                return i + 15
            item = IndicatorDataPoint(self.symbol, time, value)
            self.window.Add(item)

        @property
        def Return(self):
            if False:
                print('Hello World!')
            return pd.Series(data=[x.Value for x in self.window], index=[x.EndTime for x in self.window])

        @property
        def IsReady(self):
            if False:
                return 10
            return self.window.IsReady

        def __str__(self, **kwargs):
            if False:
                while True:
                    i = 10
            return '{}: {:.2%}'.format(self.roc.Name, self.window[0])