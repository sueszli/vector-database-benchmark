from AlgorithmImports import *

class ScheduledEventsAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity('SPY')
        self.Schedule.On(self.DateRules.On(2013, 10, 7), self.TimeRules.At(13, 0), self.SpecificTime)
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.AfterMarketOpen('SPY', 10), self.EveryDayAfterMarketOpen)
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.BeforeMarketClose('SPY', 10), self.EveryDayAfterMarketClose)
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Wednesday), self.TimeRules.At(12, 0), self.EveryWedAtNoon)
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Friday), self.TimeRules.At(12, 0), self.EveryMonFriAtNoon)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=10)), self.LiquidateUnrealizedLosses)
        self.Schedule.On(self.DateRules.MonthStart('SPY'), self.TimeRules.AfterMarketOpen('SPY'), self.RebalancingCode)

    def OnData(self, data):
        if False:
            print('Hello World!')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1)

    def SpecificTime(self):
        if False:
            print('Hello World!')
        self.Log(f'SpecificTime: Fired at : {self.Time}')

    def EveryDayAfterMarketOpen(self):
        if False:
            i = 10
            return i + 15
        self.Log(f'EveryDay.SPY 10 min after open: Fired at: {self.Time}')

    def EveryDayAfterMarketClose(self):
        if False:
            while True:
                i = 10
        self.Log(f'EveryDay.SPY 10 min before close: Fired at: {self.Time}')

    def EveryWedAtNoon(self):
        if False:
            return 10
        self.Log(f'Wed at 12pm: Fired at: {self.Time}')

    def EveryMonFriAtNoon(self):
        if False:
            return 10
        self.Log(f'Mon/Fri at 12pm: Fired at: {self.Time}')

    def LiquidateUnrealizedLosses(self):
        if False:
            while True:
                i = 10
        ' if we have over 1000 dollars in unrealized losses, liquidate'
        if self.Portfolio.TotalUnrealizedProfit < -1000:
            self.Log(f'Liquidated due to unrealized losses at: {self.Time}')
            self.Liquidate()

    def RebalancingCode(self):
        if False:
            while True:
                i = 10
        ' Good spot for rebalancing code?'
        pass