from AlgorithmImports import *

class RebalancingLeveragedETFAlpha(QCAlgorithm):
    """ Alpha Streams: Benchmark Alpha: Leveraged ETF Rebalancing
        Strategy by Prof. Shum, reposted by Ernie Chan.
        Source: http://epchan.blogspot.com/2012/10/a-leveraged-etfs-strategy.html"""

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2017, 6, 1)
        self.SetEndDate(2018, 8, 1)
        self.SetCash(100000)
        underlying = ['SPY', 'QLD', 'DIA', 'IJR', 'MDY', 'IWM', 'QQQ', 'IYE', 'EEM', 'IYW', 'EFA', 'GAZB', 'SLV', 'IEF', 'IYM', 'IYF', 'IYH', 'IYR', 'IYC', 'IBB', 'FEZ', 'USO', 'TLT']
        ultraLong = ['SSO', 'UGL', 'DDM', 'SAA', 'MZZ', 'UWM', 'QLD', 'DIG', 'EET', 'ROM', 'EFO', 'BOIL', 'AGQ', 'UST', 'UYM', 'UYG', 'RXL', 'URE', 'UCC', 'BIB', 'ULE', 'UCO', 'UBT']
        ultraShort = ['SDS', 'GLL', 'DXD', 'SDD', 'MVV', 'TWM', 'QID', 'DUG', 'EEV', 'REW', 'EFU', 'KOLD', 'ZSL', 'PST', 'SMN', 'SKF', 'RXD', 'SRS', 'SCC', 'BIS', 'EPV', 'SCO', 'TBT']
        groups = []
        for i in range(len(underlying)):
            group = ETFGroup(self.AddEquity(underlying[i], Resolution.Minute).Symbol, self.AddEquity(ultraLong[i], Resolution.Minute).Symbol, self.AddEquity(ultraShort[i], Resolution.Minute).Symbol)
            groups.append(group)
        self.SetUniverseSelection(ManualUniverseSelectionModel())
        self.SetAlpha(RebalancingLeveragedETFAlphaModel(groups))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

class RebalancingLeveragedETFAlphaModel(AlphaModel):
    """
        If the underlying ETF has experienced a return >= 1% since the previous day's close up to the current time at 14:15,
        then buy it's ultra ETF right away, and exit at the close. If the return is <= -1%, sell it's ultra-short ETF.
    """

    def __init__(self, ETFgroups):
        if False:
            print('Hello World!')
        self.ETFgroups = ETFgroups
        self.date = datetime.min.date
        self.Name = 'RebalancingLeveragedETFAlphaModel'

    def Update(self, algorithm, data):
        if False:
            return 10
        'Scan to see if the returns are greater than 1% at 2.15pm to emit an insight.'
        insights = []
        magnitude = 0.0005
        period = timedelta(minutes=105)
        if algorithm.Time.date() != self.date:
            self.date = algorithm.Time.date()
            for group in self.ETFgroups:
                history = algorithm.History([group.underlying], 1, Resolution.Daily)
                group.yesterdayClose = None if history.empty else history.loc[str(group.underlying)]['close'][0]
        if algorithm.Time.hour == 14 and algorithm.Time.minute == 15:
            for group in self.ETFgroups:
                if group.yesterdayClose == 0 or group.yesterdayClose is None:
                    continue
                returns = round((algorithm.Portfolio[group.underlying].Price - group.yesterdayClose) / group.yesterdayClose, 10)
                if returns > 0.01:
                    insights.append(Insight.Price(group.ultraLong, period, InsightDirection.Up, magnitude))
                elif returns < -0.01:
                    insights.append(Insight.Price(group.ultraShort, period, InsightDirection.Down, magnitude))
        return insights

class ETFGroup:
    """
    Group the underlying ETF and it's ultra ETFs
    Args:
        underlying: The underlying index ETF
        ultraLong: The long-leveraged version of underlying ETF
        ultraShort: The short-leveraged version of the underlying ETF
    """

    def __init__(self, underlying, ultraLong, ultraShort):
        if False:
            for i in range(10):
                print('nop')
        self.underlying = underlying
        self.ultraLong = ultraLong
        self.ultraShort = ultraShort
        self.yesterdayClose = 0