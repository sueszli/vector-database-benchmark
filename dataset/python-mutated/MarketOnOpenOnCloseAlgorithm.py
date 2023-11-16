from AlgorithmImports import *

class MarketOnOpenOnCloseAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.equity = self.AddEquity('SPY', Resolution.Second, fillForward=True, extendedMarketHours=True)
        self.__submittedMarketOnCloseToday = False
        self.__last = datetime.min

    def OnData(self, data):
        if False:
            return 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if self.Time.date() != self.__last.date():
            self.__submittedMarketOnCloseToday = False
            self.MarketOnOpenOrder('SPY', 100)
            self.__last = self.Time
        if not self.__submittedMarketOnCloseToday and self.equity.Exchange.ExchangeOpen:
            self.__submittedMarketOnCloseToday = True
            self.MarketOnCloseOrder('SPY', -100)

    def OnOrderEvent(self, fill):
        if False:
            return 10
        order = self.Transactions.GetOrderById(fill.OrderId)
        self.Log('{0} - {1}:: {2}'.format(self.Time, order.Type, fill))