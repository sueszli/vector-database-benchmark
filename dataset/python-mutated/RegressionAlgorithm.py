from AlgorithmImports import *

class RegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(10000000)
        self.AddEquity('SPY', Resolution.Tick)
        self.AddEquity('BAC', Resolution.Minute)
        self.AddEquity('AIG', Resolution.Hour)
        self.AddEquity('IBM', Resolution.Daily)
        self.__lastTradeTicks = self.StartDate
        self.__lastTradeTradeBars = self.__lastTradeTicks
        self.__tradeEvery = timedelta(minutes=1)

    def OnData(self, data):
        if False:
            return 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if self.Time - self.__lastTradeTradeBars < self.__tradeEvery:
            return
        self.__lastTradeTradeBars = self.Time
        for kvp in data.Bars:
            period = kvp.Value.Period.total_seconds()
            if self.roundTime(self.Time, period) != self.Time:
                pass
            symbol = kvp.Key
            holdings = self.Portfolio[symbol]
            if not holdings.Invested:
                self.MarketOrder(symbol, 10)
            else:
                self.MarketOrder(symbol, -holdings.Quantity)

    def roundTime(self, dt=None, roundTo=60):
        if False:
            while True:
                i = 10
        'Round a datetime object to any time laps in seconds\n        dt : datetime object, default now.\n        roundTo : Closest number of seconds to round to, default 1 minute.\n        '
        if dt is None:
            dt = datetime.now()
        seconds = (dt - dt.min).seconds
        rounding = (seconds + roundTo / 2) // roundTo * roundTo
        return dt + timedelta(0, rounding - seconds, -dt.microsecond)