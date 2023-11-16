from AlgorithmImports import *

class DataConsolidationAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(DateTime(2013, 10, 7, 9, 30, 0))
        self.SetEndDate(self.StartDate + timedelta(60))
        self.AddEquity('SPY')
        self.AddForex('EURUSD', Resolution.Hour)
        thirtyMinuteConsolidator = TradeBarConsolidator(timedelta(minutes=30))
        thirtyMinuteConsolidator.DataConsolidated += self.ThirtyMinuteBarHandler
        self.SubscriptionManager.AddConsolidator('SPY', thirtyMinuteConsolidator)
        oneDayConsolidator = TradeBarConsolidator(timedelta(1))
        threeCountConsolidator = TradeBarConsolidator(3)
        three_oneDayBar = SequentialConsolidator(oneDayConsolidator, threeCountConsolidator)
        three_oneDayBar.DataConsolidated += self.ThreeDayBarConsolidatedHandler
        self.SubscriptionManager.AddConsolidator('SPY', three_oneDayBar)
        customMonthlyConsolidator = TradeBarConsolidator(self.CustomMonthly)
        customMonthlyConsolidator.DataConsolidated += self.CustomMonthlyHandler
        self.SubscriptionManager.AddConsolidator('SPY', customMonthlyConsolidator)
        self.Consolidate('SPY', timedelta(minutes=45), self.FortyFiveMinuteBarHandler)
        self.Consolidate('SPY', Resolution.Hour, self.HourBarHandler)
        self.Consolidate('EURUSD', Resolution.Daily, self.DailyEurUsdBarHandler)
        self.Consolidate('SPY', Calendar.Weekly, self.CalendarTradeBarHandler)
        self.Consolidate('EURUSD', Calendar.Weekly, self.CalendarQuoteBarHandler)
        self.Consolidate('SPY', Calendar.Monthly, self.CalendarTradeBarHandler)
        self.Consolidate('EURUSD', Calendar.Monthly, self.CalendarQuoteBarHandler)
        self.Consolidate('SPY', Calendar.Quarterly, self.CalendarTradeBarHandler)
        self.Consolidate('EURUSD', Calendar.Quarterly, self.CalendarQuoteBarHandler)
        self.Consolidate('SPY', Calendar.Yearly, self.CalendarTradeBarHandler)
        self.Consolidate('EURUSD', Calendar.Yearly, self.CalendarQuoteBarHandler)
        self.consolidatedHour = False
        self.consolidated45Minute = False
        self.__last = None

    def OnData(self, data):
        if False:
            while True:
                i = 10
        'We need to declare this method'
        pass

    def OnEndOfDay(self):
        if False:
            print('Hello World!')
        self.Liquidate('SPY')
        self.__last = None

    def ThirtyMinuteBarHandler(self, sender, consolidated):
        if False:
            while True:
                i = 10
        "This is our event handler for our 30 minute trade bar defined above in Initialize(). So each time the\n        consolidator produces a new 30 minute bar, this function will be called automatically. The 'sender' parameter\n         will be the instance of the IDataConsolidator that invoked the event, but you'll almost never need that!"
        if self.__last is not None and consolidated.Close > self.__last.Close:
            self.Log(f"{consolidated.Time} >> SPY >> LONG  >> 100 >> {self.Portfolio['SPY'].Quantity}")
            self.Order('SPY', 100)
        elif self.__last is not None and consolidated.Close < self.__last.Close:
            self.Log(f"{consolidated.Time} >> SPY >> SHORT  >> 100 >> {self.Portfolio['SPY'].Quantity}")
            self.Order('SPY', -100)
        self.__last = consolidated

    def ThreeDayBarConsolidatedHandler(self, sender, consolidated):
        if False:
            return 10
        " This is our event handler for our 3 day trade bar defined above in Initialize(). So each time the\n        consolidator produces a new 3 day bar, this function will be called automatically. The 'sender' parameter\n        will be the instance of the IDataConsolidator that invoked the event, but you'll almost never need that!"
        self.Log(f'{consolidated.Time} >> Plotting!')
        self.Plot(consolidated.Symbol.Value, '3HourBar', consolidated.Close)

    def FortyFiveMinuteBarHandler(self, consolidated):
        if False:
            return 10
        ' This is our event handler for our 45 minute consolidated defined using the Consolidate method'
        self.consolidated45Minute = True
        self.Log(f'{consolidated.EndTime} >> FortyFiveMinuteBarHandler >> {consolidated.Close}')

    def HourBarHandler(self, consolidated):
        if False:
            for i in range(10):
                print('nop')
        'This is our event handler for our one hour consolidated defined using the Consolidate method'
        self.consolidatedHour = True
        self.Log(f'{consolidated.EndTime} >> FortyFiveMinuteBarHandler >> {consolidated.Close}')

    def DailyEurUsdBarHandler(self, consolidated):
        if False:
            print('Hello World!')
        'This is our event handler for our daily consolidated defined using the Consolidate method'
        self.Log(f'{consolidated.EndTime} EURUSD Daily consolidated.')

    def CalendarTradeBarHandler(self, tradeBar):
        if False:
            i = 10
            return i + 15
        self.Log(f'{self.Time} :: {tradeBar.Time} {tradeBar.Close}')

    def CalendarQuoteBarHandler(self, quoteBar):
        if False:
            for i in range(10):
                print('nop')
        self.Log(f'{self.Time} :: {quoteBar.Time} {quoteBar.Close}')

    def CustomMonthly(self, dt):
        if False:
            print('Hello World!')
        'Custom Monthly Func'
        start = dt.replace(day=1).date()
        end = dt.replace(day=28) + timedelta(4)
        end = (end - timedelta(end.day - 1)).date()
        return CalendarInfo(start, end - start)

    def CustomMonthlyHandler(self, sender, consolidated):
        if False:
            print('Hello World!')
        'This is our event handler Custom Monthly function'
        self.Log(f'{consolidated.Time} >> CustomMonthlyHandler >> {consolidated.Close}')

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        if not self.consolidatedHour:
            raise Exception('Expected hourly consolidator to be fired.')
        if not self.consolidated45Minute:
            raise Exception('Expected 45-minute consolidator to be fired.')