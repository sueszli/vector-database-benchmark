from AlgorithmImports import *
from Portfolio.RiskParityPortfolioOptimizer import RiskParityPortfolioOptimizer

class RiskParityPortfolioConstructionModel(PortfolioConstructionModel):

    def __init__(self, rebalance=Resolution.Daily, portfolioBias=PortfolioBias.LongShort, lookback=1, period=252, resolution=Resolution.Daily, optimizer=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the model\n        Args:\n            rebalance: Rebalancing parameter. If it is a timedelta, date rules or Resolution, it will be converted into a function.\n                              If None will be ignored.\n                              The function returns the next expected rebalance time for a given algorithm UTC DateTime.\n                              The function returns null if unknown, in which case the function will be called again in the\n                              next loop. Returning current time will trigger rebalance.\n            portfolioBias: Specifies the bias of the portfolio (Short, Long/Short, Long)\n            lookback(int): Historical return lookback period\n            period(int): The time interval of history price to calculate the weight\n            resolution: The resolution of the history price\n            optimizer(class): Method used to compute the portfolio weights'
        super().__init__()
        if portfolioBias == PortfolioBias.Short:
            raise ArgumentException('Long position must be allowed in RiskParityPortfolioConstructionModel.')
        self.lookback = lookback
        self.period = period
        self.resolution = resolution
        self.sign = lambda x: -1 if x < 0 else 1 if x > 0 else 0
        self.optimizer = RiskParityPortfolioOptimizer() if optimizer is None else optimizer
        self.symbolDataBySymbol = {}
        rebalancingFunc = rebalance
        if isinstance(rebalance, int):
            rebalance = Extensions.ToTimeSpan(rebalance)
        if isinstance(rebalance, timedelta):
            rebalancingFunc = lambda dt: dt + rebalance
        if rebalancingFunc:
            self.SetRebalancingFunc(rebalancingFunc)

    def DetermineTargetPercent(self, activeInsights):
        if False:
            while True:
                i = 10
        'Will determine the target percent for each insight\n        Args:\n            activeInsights: list of active insights\n        Returns:\n            dictionary of insight and respective target weight\n        '
        targets = {}
        if len(activeInsights) == 0:
            return targets
        symbols = [insight.Symbol for insight in activeInsights]
        returns = {str(symbol): data.Return for (symbol, data) in self.symbolDataBySymbol.items() if symbol in symbols}
        returns = pd.DataFrame(returns)
        weights = self.optimizer.Optimize(returns)
        weights = pd.Series(weights, index=returns.columns)
        for insight in activeInsights:
            targets[insight] = weights[str(insight.Symbol)]
        return targets

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            for i in range(10):
                print('nop')
        'Event fired each time the we add/remove securities from the data feed\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        super().OnSecuritiesChanged(algorithm, changes)
        for removed in changes.RemovedSecurities:
            symbolData = self.symbolDataBySymbol.pop(removed.Symbol, None)
            symbolData.Reset()
            algorithm.UnregisterIndicator(symbolData.roc)
        symbols = [x.Symbol for x in changes.AddedSecurities]
        history = algorithm.History(symbols, self.lookback * self.period, self.resolution)
        if history.empty:
            return
        tickers = history.index.levels[0]
        for ticker in tickers:
            symbol = SymbolCache.GetSymbol(ticker)
            if symbol not in self.symbolDataBySymbol:
                symbolData = self.RiskParitySymbolData(symbol, self.lookback, self.period)
                symbolData.WarmUpIndicators(history.loc[ticker])
                self.symbolDataBySymbol[symbol] = symbolData
                algorithm.RegisterIndicator(symbol, symbolData.roc, self.resolution)

    class RiskParitySymbolData:
        """Contains data specific to a symbol required by this model"""

        def __init__(self, symbol, lookback, period):
            if False:
                for i in range(10):
                    print('nop')
            self.symbol = symbol
            self.roc = RateOfChange(f'{symbol}.ROC({lookback})', lookback)
            self.roc.Updated += self.OnRateOfChangeUpdated
            self.window = RollingWindow[IndicatorDataPoint](period)

        def Reset(self):
            if False:
                for i in range(10):
                    print('nop')
            self.roc.Updated -= self.OnRateOfChangeUpdated
            self.roc.Reset()
            self.window.Reset()

        def WarmUpIndicators(self, history):
            if False:
                for i in range(10):
                    print('nop')
            for tuple in history.itertuples():
                self.roc.Update(tuple.Index, tuple.close)

        def OnRateOfChangeUpdated(self, roc, value):
            if False:
                return 10
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
                return 10
            return pd.Series(data=[x.Value for x in self.window], index=[x.EndTime for x in self.window])

        @property
        def IsReady(self):
            if False:
                print('Hello World!')
            return self.window.IsReady

        def __str__(self, **kwargs):
            if False:
                i = 10
                return i + 15
            return '{}: {:.2%}'.format(self.roc.Name, self.window[0])