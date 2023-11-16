from AlgorithmImports import *
from Portfolio.MaximumSharpeRatioPortfolioOptimizer import MaximumSharpeRatioPortfolioOptimizer
from itertools import groupby
from numpy import dot, transpose
from numpy.linalg import inv

class BlackLittermanOptimizationPortfolioConstructionModel(PortfolioConstructionModel):

    def __init__(self, rebalance=Resolution.Daily, portfolioBias=PortfolioBias.LongShort, lookback=1, period=63, resolution=Resolution.Daily, risk_free_rate=0, delta=2.5, tau=0.05, optimizer=None):
        if False:
            print('Hello World!')
        'Initialize the model\n        Args:\n            rebalance: Rebalancing parameter. If it is a timedelta, date rules or Resolution, it will be converted into a function.\n                              If None will be ignored.\n                              The function returns the next expected rebalance time for a given algorithm UTC DateTime.\n                              The function returns null if unknown, in which case the function will be called again in the\n                              next loop. Returning current time will trigger rebalance.\n            portfolioBias: Specifies the bias of the portfolio (Short, Long/Short, Long)\n            lookback(int): Historical return lookback period\n            period(int): The time interval of history price to calculate the weight\n            resolution: The resolution of the history price\n            risk_free_rate(float): The risk free rate\n            delta(float): The risk aversion coeffficient of the market portfolio\n            tau(float): The model parameter indicating the uncertainty of the CAPM prior'
        super().__init__()
        self.lookback = lookback
        self.period = period
        self.resolution = resolution
        self.risk_free_rate = risk_free_rate
        self.delta = delta
        self.tau = tau
        self.portfolioBias = portfolioBias
        lower = 0 if portfolioBias == PortfolioBias.Long else -1
        upper = 0 if portfolioBias == PortfolioBias.Short else 1
        self.optimizer = MaximumSharpeRatioPortfolioOptimizer(lower, upper, risk_free_rate) if optimizer is None else optimizer
        self.sign = lambda x: -1 if x < 0 else 1 if x > 0 else 0
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
            print('Hello World!')
        return PortfolioConstructionModel.FilterInvalidInsightMagnitude(self.Algorithm, [insight])

    def DetermineTargetPercent(self, lastActiveInsights):
        if False:
            return 10
        targets = {}
        (P, Q) = self.get_views(lastActiveInsights)
        if P is not None:
            returns = dict()
            for insight in lastActiveInsights:
                symbol = insight.Symbol
                symbolData = self.symbolDataBySymbol.get(symbol, self.BlackLittermanSymbolData(symbol, self.lookback, self.period))
                if insight.Magnitude is None:
                    self.Algorithm.SetRunTimeError(ArgumentNullException("BlackLittermanOptimizationPortfolioConstructionModel does not accept 'None' as Insight.Magnitude. Please make sure your Alpha Model is generating Insights with the Magnitude property set."))
                    return targets
                symbolData.Add(insight.GeneratedTimeUtc, insight.Magnitude)
                returns[symbol] = symbolData.Return
            returns = pd.DataFrame(returns)
            (Pi, Sigma) = self.get_equilibrium_return(returns)
            (Pi, Sigma) = self.apply_blacklitterman_master_formula(Pi, Sigma, P, Q)
            weights = self.optimizer.Optimize(returns, Pi, Sigma)
            weights = pd.Series(weights, index=Sigma.columns)
            for (symbol, weight) in weights.items():
                for insight in lastActiveInsights:
                    if str(insight.Symbol) == str(symbol):
                        if self.portfolioBias != PortfolioBias.LongShort and self.sign(weight) != self.portfolioBias:
                            weight = 0
                        targets[insight] = weight
                        break
        return targets

    def GetTargetInsights(self):
        if False:
            for i in range(10):
                print('nop')
        activeInsights = filter(self.ShouldCreateTargetForInsight, self.Algorithm.Insights.GetActiveInsights(self.Algorithm.UtcTime))
        lastActiveInsights = []
        for (sourceModel, f) in groupby(sorted(activeInsights, key=lambda ff: ff.SourceModel), lambda fff: fff.SourceModel):
            for (symbol, g) in groupby(sorted(list(f), key=lambda gg: gg.Symbol), lambda ggg: ggg.Symbol):
                lastActiveInsights.append(sorted(g, key=lambda x: x.GeneratedTimeUtc)[-1])
        return lastActiveInsights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            for i in range(10):
                print('nop')
        'Event fired each time the we add/remove securities from the data feed\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        super().OnSecuritiesChanged(algorithm, changes)
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            symbolData = self.symbolDataBySymbol.pop(symbol, None)
            if symbolData is not None:
                symbolData.Reset()
        addedSymbols = {x.Symbol: x.Exchange.TimeZone for x in changes.AddedSecurities}
        history = algorithm.History(list(addedSymbols.keys()), self.lookback * self.period, self.resolution)
        if history.empty:
            return
        history = history.close.unstack(0)
        symbols = history.columns
        for (symbol, timezone) in addedSymbols.items():
            if str(symbol) not in symbols:
                continue
            symbolData = self.symbolDataBySymbol.get(symbol, self.BlackLittermanSymbolData(symbol, self.lookback, self.period))
            for (time, close) in history[symbol].items():
                utcTime = Extensions.ConvertToUtc(time, timezone)
                symbolData.Update(utcTime, close)
            self.symbolDataBySymbol[symbol] = symbolData

    def apply_blacklitterman_master_formula(self, Pi, Sigma, P, Q):
        if False:
            for i in range(10):
                print('nop')
        'Apply Black-Litterman master formula\n        http://www.blacklitterman.org/cookbook.html\n        Args:\n            Pi: Prior/Posterior mean array\n            Sigma: Prior/Posterior covariance matrix\n            P: A matrix that identifies the assets involved in the views (size: K x N)\n            Q: A view vector (size: K x 1)'
        ts = self.tau * Sigma
        omega = np.dot(np.dot(P, ts), P.T) * np.eye(Q.shape[0])
        if np.linalg.det(omega) == 0:
            return (Pi, Sigma)
        A = np.dot(np.dot(ts, P.T), inv(np.dot(np.dot(P, ts), P.T) + omega))
        Pi = np.squeeze(np.asarray(np.expand_dims(Pi, axis=0).T + np.dot(A, Q - np.expand_dims(np.dot(P, Pi.T), axis=1))))
        M = ts - np.dot(np.dot(A, P), ts)
        Sigma = (Sigma + M) * self.delta
        return (Pi, Sigma)

    def get_equilibrium_return(self, returns):
        if False:
            i = 10
            return i + 15
        'Calculate equilibrium returns and covariance\n        Args:\n            returns: Matrix of returns where each column represents a security and each row returns for the given date/time (size: K x N)\n        Returns:\n            equilibrium_return: Array of double of equilibrium returns\n            cov: Multi-dimensional array of double with the portfolio covariance of returns (size: K x K)'
        size = len(returns.columns)
        W = np.array([1 / size] * size)
        cov = returns.cov() * 252
        annual_return = np.sum(((1 + returns.mean()) ** 252 - 1) * W)
        annual_variance = dot(W.T, dot(cov, W))
        risk_aversion = (annual_return - self.risk_free_rate) / annual_variance
        equilibrium_return = dot(dot(risk_aversion, cov), W)
        return (equilibrium_return, cov)

    def get_views(self, insights):
        if False:
            while True:
                i = 10
        "Generate views from multiple alpha models\n        Args\n            insights: Array of insight that represent the investors' views\n        Returns\n            P: A matrix that identifies the assets involved in the views (size: K x N)\n            Q: A view vector (size: K x 1)"
        try:
            P = {}
            Q = {}
            symbols = set((insight.Symbol for insight in insights))
            for (model, group) in groupby(insights, lambda x: x.SourceModel):
                group = list(group)
                up_insights_sum = 0.0
                dn_insights_sum = 0.0
                for insight in group:
                    if insight.Direction == InsightDirection.Up:
                        up_insights_sum = up_insights_sum + np.abs(insight.Magnitude)
                    if insight.Direction == InsightDirection.Down:
                        dn_insights_sum = dn_insights_sum + np.abs(insight.Magnitude)
                q = up_insights_sum if up_insights_sum > dn_insights_sum else dn_insights_sum
                if q == 0:
                    continue
                Q[model] = q
                P[model] = dict()
                for insight in group:
                    value = insight.Direction * np.abs(insight.Magnitude)
                    P[model][insight.Symbol] = value / q
                for symbol in symbols:
                    if symbol not in P[model]:
                        P[model][symbol] = 0
            Q = np.array([[x] for x in Q.values()])
            if len(Q) > 0:
                P = np.array([list(x.values()) for x in P.values()])
                return (P, Q)
        except:
            pass
        return (None, None)

    class BlackLittermanSymbolData:
        """Contains data specific to a symbol required by this model"""

        def __init__(self, symbol, lookback, period):
            if False:
                return 10
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

        def Update(self, utcTime, close):
            if False:
                i = 10
                return i + 15
            self.roc.Update(utcTime, close)

        def OnRateOfChangeUpdated(self, roc, value):
            if False:
                for i in range(10):
                    print('nop')
            if roc.IsReady:
                self.window.Add(value)

        def Add(self, time, value):
            if False:
                while True:
                    i = 10
            if self.window.Samples > 0 and self.window[0].EndTime == time:
                return
            item = IndicatorDataPoint(self.symbol, time, value)
            self.window.Add(item)

        @property
        def Return(self):
            if False:
                for i in range(10):
                    print('nop')
            return pd.Series(data=[x.Value for x in self.window], index=[x.EndTime for x in self.window])

        @property
        def IsReady(self):
            if False:
                while True:
                    i = 10
            return self.window.IsReady

        def __str__(self, **kwargs):
            if False:
                return 10
            return f'{self.roc.Name}: {(1 + self.window[0]) ** 252 - 1:.2%}'