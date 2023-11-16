from AlgorithmImports import *

class MeanReversionPortfolioConstructionModel(PortfolioConstructionModel):

    def __init__(self, rebalance=Resolution.Daily, portfolioBias=PortfolioBias.LongShort, reversion_threshold=1, window_size=20, resolution=Resolution.Daily):
        if False:
            while True:
                i = 10
        'Initialize the model\n        Args:\n            rebalance: Rebalancing parameter. If it is a timedelta, date rules or Resolution, it will be converted into a function.\n                              If None will be ignored.\n                              The function returns the next expected rebalance time for a given algorithm UTC DateTime.\n                              The function returns null if unknown, in which case the function will be called again in the\n                              next loop. Returning current time will trigger rebalance.\n            portfolioBias: Specifies the bias of the portfolio (Short, Long/Short, Long)\n            reversion_threshold: Reversion threshold\n            window_size: Window size of mean price calculation\n            resolution: The resolution of the history price and rebalancing\n        '
        super().__init__()
        if portfolioBias == PortfolioBias.Short:
            raise ArgumentException('Long position must be allowed in MeanReversionPortfolioConstructionModel.')
        self.reversion_threshold = reversion_threshold
        self.window_size = window_size
        self.resolution = resolution
        self.num_of_assets = 0
        self.symbol_data = {}
        rebalancingFunc = rebalance
        if isinstance(rebalance, int):
            rebalance = Extensions.ToTimeSpan(rebalance)
        if isinstance(rebalance, timedelta):
            rebalancingFunc = lambda dt: dt + rebalance
        if rebalancingFunc:
            self.SetRebalancingFunc(rebalancingFunc)

    def DetermineTargetPercent(self, activeInsights):
        if False:
            for i in range(10):
                print('nop')
        'Will determine the target percent for each insight\n        Args:\n            activeInsights: list of active insights\n        Returns:\n            dictionary of insight and respective target weight\n        '
        targets = {}
        if len(activeInsights) == 0 or not all([self.symbol_data[x.Symbol].IsReady for x in activeInsights]):
            return targets
        num_of_assets = len(activeInsights)
        if self.num_of_assets != num_of_assets:
            self.num_of_assets = num_of_assets
            self.weight_vector = np.ones(num_of_assets) * (1 / num_of_assets)
        price_relatives = self.GetPriceRelatives(activeInsights)
        next_prediction = price_relatives.mean()
        assets_mean_dev = price_relatives - next_prediction
        second_norm = np.linalg.norm(assets_mean_dev) ** 2
        if second_norm == 0.0:
            step_size = 0
        else:
            step_size = (np.dot(self.weight_vector, price_relatives) - self.reversion_threshold) / second_norm
            step_size = max(0, step_size)
        next_portfolio = self.weight_vector - step_size * assets_mean_dev
        normalized_portfolio_weight_vector = self.SimplexProjection(next_portfolio)
        self.weight_vector = normalized_portfolio_weight_vector
        for (i, insight) in enumerate(activeInsights):
            targets[insight] = normalized_portfolio_weight_vector[i]
        return targets

    def GetPriceRelatives(self, activeInsights):
        if False:
            for i in range(10):
                print('nop')
        'Get price relatives with reference level of SMA\n        Args:\n            activeInsights: list of active insights\n        Returns:\n            array of price relatives vector\n        '
        next_price_relatives = np.zeros(len(activeInsights))
        for (i, insight) in enumerate(activeInsights):
            symbol_data = self.symbol_data[insight.Symbol]
            next_price_relatives[i] = 1 + insight.Magnitude * insight.Direction if insight.Magnitude is not None else symbol_data.Identity.Current.Value / symbol_data.Sma.Current.Value
        return next_price_relatives

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            while True:
                i = 10
        'Event fired each time the we add/remove securities from the data feed\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm\n        '
        super().OnSecuritiesChanged(algorithm, changes)
        for removed in changes.RemovedSecurities:
            symbol_data = self.symbol_data.pop(removed.Symbol, None)
            symbol_data.Reset()
        symbols = [x.Symbol for x in changes.AddedSecurities]
        for symbol in symbols:
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = self.MeanReversionSymbolData(algorithm, symbol, self.window_size, self.resolution)

    def SimplexProjection(self, vector, total=1):
        if False:
            for i in range(10):
                print('nop')
        'Normalize the updated portfolio into weight vector:\n        v_{t+1} = arg min || v - v_{t+1} || ^ 2\n        Implementation from:\n        Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008, July). \n            Efficient projections onto the l 1-ball for learning in high dimensions.\n            In Proceedings of the 25th international conference on Machine learning \n            (pp. 272-279).\n        Args:\n            vector: unnormalized weight vector\n            total: total weight of output, default to be 1, making it a probabilistic simplex\n        '
        if total <= 0:
            raise ArgumentException('Total must be > 0 for Euclidean Projection onto the Simplex.')
        vector = np.asarray(vector)
        mu = np.sort(vector)[::-1]
        sv = np.cumsum(mu)
        rho = np.where(mu > (sv - total) / np.arange(1, len(vector) + 1))[0][-1]
        theta = (sv[rho] - total) / (rho + 1)
        w = vector - theta
        w[w < 0] = 0
        return w

    class MeanReversionSymbolData:

        def __init__(self, algo, symbol, window_size, resolution):
            if False:
                print('Hello World!')
            self.Identity = algo.Identity(symbol, resolution)
            self.Sma = algo.SMA(symbol, window_size, resolution)
            algo.WarmUpIndicator(symbol, self.Identity, resolution)
            algo.WarmUpIndicator(symbol, self.Sma, resolution)

        def Reset(self):
            if False:
                while True:
                    i = 10
            self.Identity.Reset()
            self.Sma.Reset()

        @property
        def IsReady(self):
            if False:
                while True:
                    i = 10
            return self.Identity.IsReady and self.Sma.IsReady