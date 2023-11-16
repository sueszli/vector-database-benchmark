from AlgorithmImports import *
from sklearn.linear_model import LinearRegression

class ScikitLearnLinearRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 8)
        self.lookback = 30
        self.SetCash(100000)
        spy = self.AddEquity('SPY', Resolution.Minute)
        self.symbols = [spy.Symbol]
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.AfterMarketOpen('SPY', 28), self.Regression)
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.AfterMarketOpen('SPY', 30), self.Trade)

    def Regression(self):
        if False:
            while True:
                i = 10
        history = self.History(self.symbols, self.lookback, Resolution.Daily)
        self.prices = {}
        self.slopes = {}
        for symbol in self.symbols:
            if not history.empty:
                self.prices[symbol] = list(history.loc[symbol.Value]['open'])
        A = range(self.lookback + 1)
        for symbol in self.symbols:
            if symbol in self.prices:
                Y = self.prices[symbol]
                X = np.column_stack([np.ones(len(A)), A])
                length = min(len(X), len(Y))
                X = X[-length:]
                Y = Y[-length:]
                A = A[-length:]
                reg = LinearRegression().fit(X, Y)
                b = reg.intercept_
                a = reg.coef_[1]
                self.slopes[symbol] = a / b

    def Trade(self):
        if False:
            return 10
        if not self.prices:
            return
        thod_buy = 0.001
        thod_liquidate = -0.001
        for holding in self.Portfolio.Values:
            slope = self.slopes[holding.Symbol]
            if holding.Invested and slope < thod_liquidate:
                self.Liquidate(holding.Symbol)
        for symbol in self.symbols:
            if self.slopes[symbol] > thod_buy:
                self.SetHoldings(symbol, 1 / len(self.symbols))