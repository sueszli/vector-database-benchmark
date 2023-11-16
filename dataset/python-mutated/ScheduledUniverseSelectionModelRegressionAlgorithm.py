from AlgorithmImports import *

class ScheduledUniverseSelectionModelRegressionAlgorithm(QCAlgorithm):
    """Regression algorithm for testing ScheduledUniverseSelectionModel scheduling functions."""

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.UniverseSettings.Resolution = Resolution.Hour
        self.SetStartDate(2017, 1, 1)
        self.SetEndDate(2017, 2, 1)
        self.SetUniverseSelection(ScheduledUniverseSelectionModel(self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Thursday), self.TimeRules.Every(timedelta(hours=12)), self.SelectSymbols))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(1)))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.seenDays = []

    def SelectSymbols(self, dateTime):
        if False:
            while True:
                i = 10
        symbols = []
        weekday = dateTime.weekday()
        if weekday == 0 or weekday == 1:
            symbols.append(Symbol.Create('SPY', SecurityType.Equity, Market.USA))
        elif weekday == 2:
            symbols.append(Symbol.Create('AAPL', SecurityType.Equity, Market.USA))
        else:
            symbols.append(Symbol.Create('IBM', SecurityType.Equity, Market.USA))
        if weekday == 1 or weekday == 3:
            symbols.append(Symbol.Create('EURUSD', SecurityType.Forex, Market.Oanda))
        elif weekday == 4:
            symbols.append(Symbol.Create('EURGBP', SecurityType.Forex, Market.Oanda))
        else:
            symbols.append(Symbol.Create('NZDUSD', SecurityType.Forex, Market.Oanda))
        return symbols

    def OnSecuritiesChanged(self, changes):
        if False:
            while True:
                i = 10
        self.Log('{}: {}'.format(self.Time, changes))
        weekday = self.Time.weekday()
        if weekday == 0:
            self.ExpectAdditions(changes, 'SPY', 'NZDUSD')
            if weekday not in self.seenDays:
                self.seenDays.append(weekday)
                self.ExpectRemovals(changes, None)
            else:
                self.ExpectRemovals(changes, 'EURUSD', 'IBM')
        if weekday == 1:
            self.ExpectAdditions(changes, 'EURUSD')
            if weekday not in self.seenDays:
                self.seenDays.append(weekday)
                self.ExpectRemovals(changes, 'NZDUSD')
            else:
                self.ExpectRemovals(changes, 'NZDUSD')
        if weekday == 2 or weekday == 4:
            self.ExpectAdditions(changes, None)
            self.ExpectRemovals(changes, None)
        if weekday == 3:
            self.ExpectAdditions(changes, 'IBM')
            self.ExpectRemovals(changes, 'SPY')

    def OnOrderEvent(self, orderEvent):
        if False:
            for i in range(10):
                print('nop')
        self.Log('{}: {}'.format(self.Time, orderEvent))

    def ExpectAdditions(self, changes, *tickers):
        if False:
            print('Hello World!')
        if tickers is None and changes.AddedSecurities.Count > 0:
            raise Exception('{}: Expected no additions: {}'.format(self.Time, self.Time.weekday()))
        for ticker in tickers:
            if ticker is not None and ticker not in [s.Symbol.Value for s in changes.AddedSecurities]:
                raise Exception('{}: Expected {} to be added: {}'.format(self.Time, ticker, self.Time.weekday()))

    def ExpectRemovals(self, changes, *tickers):
        if False:
            print('Hello World!')
        if tickers is None and changes.RemovedSecurities.Count > 0:
            raise Exception('{}: Expected no removals: {}'.format(self.Time, self.Time.weekday()))
        for ticker in tickers:
            if ticker is not None and ticker not in [s.Symbol.Value for s in changes.RemovedSecurities]:
                raise Exception('{}: Expected {} to be removed: {}'.format(self.Time, ticker, self.Time.weekday()))