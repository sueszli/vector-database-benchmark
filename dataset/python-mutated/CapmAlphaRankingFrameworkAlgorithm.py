from AlgorithmImports import *

class CapmAlphaRankingFrameworkAlgorithm(QCAlgorithm):
    """CapmAlphaRankingFrameworkAlgorithm: example of custom scheduled universe selection model"""

    def Initialize(self):
        if False:
            return 10
        ' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetStartDate(2016, 1, 1)
        self.SetEndDate(2017, 1, 1)
        self.SetCash(100000)
        self.SetUniverseSelection(CapmAlphaRankingUniverseSelectionModel())
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(1), 0.025, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.01))
from QuantConnect.Data.UniverseSelection import ScheduledUniverse
from Selection.UniverseSelectionModel import UniverseSelectionModel

class CapmAlphaRankingUniverseSelectionModel(UniverseSelectionModel):
    """This universe selection model picks stocks with the highest alpha: interception of the linear regression against a benchmark."""
    period = 21
    benchmark = 'SPY'
    symbols = [Symbol.Create(x, SecurityType.Equity, Market.USA) for x in ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM']]

    def CreateUniverses(self, algorithm):
        if False:
            print('Hello World!')
        benchmark = algorithm.AddEquity(self.benchmark, Resolution.Daily)
        return [ScheduledUniverse(benchmark.Exchange.TimeZone, algorithm.DateRules.MonthStart(self.benchmark), algorithm.TimeRules.AfterMarketOpen(self.benchmark), lambda datetime: self.SelectPair(algorithm, datetime), algorithm.UniverseSettings, algorithm.SecurityInitializer)]

    def SelectPair(self, algorithm, date):
        if False:
            print('Hello World!')
        'Selects the pair (two stocks) with the highest alpha'
        dictionary = dict()
        benchmark = self._getReturns(algorithm, self.benchmark)
        ones = np.ones(len(benchmark))
        for symbol in self.symbols:
            prices = self._getReturns(algorithm, symbol)
            if prices is None:
                continue
            A = np.vstack([prices, ones]).T
            ols = np.linalg.lstsq(A, benchmark)[0]
            dictionary[symbol] = ols[1]
        orderedDictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in orderedDictionary[:2]]

    def _getReturns(self, algorithm, symbol):
        if False:
            print('Hello World!')
        history = algorithm.History([symbol], self.period, Resolution.Daily)
        if history.empty:
            return None
        window = RollingWindow[float](self.period)
        rateOfChange = RateOfChange(1)

        def roc_updated(s, item):
            if False:
                while True:
                    i = 10
            window.Add(item.Value)
        rateOfChange.Updated += roc_updated
        history = history.close.reset_index(level=0, drop=True).iteritems()
        for (time, value) in history:
            rateOfChange.Update(time, value)
        return [x for x in window]