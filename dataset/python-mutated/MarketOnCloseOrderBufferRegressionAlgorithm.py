from AlgorithmImports import *

class MarketOnCloseOrderBufferRegressionAlgorithm(QCAlgorithm):
    validOrderTicket = None
    invalidOrderTicket = None

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 8)
        self.AddEquity('SPY', Resolution.Minute)

        def mocAtPostMarket():
            if False:
                print('Hello World!')
            self.validOrderTicketExtendedMarketHours = self.MarketOnCloseOrder('SPY', 2)
        self.Schedule.On(self.DateRules.Today, self.TimeRules.At(17, 0), mocAtPostMarket)
        MarketOnCloseOrder.SubmissionTimeBuffer = timedelta(minutes=10)

    def OnData(self, data):
        if False:
            return 10
        if self.Time.hour == 15 and self.Time.minute == 49 and (not self.validOrderTicket):
            self.validOrderTicket = self.MarketOnCloseOrder('SPY', 2)
        if self.Time.hour == 15 and self.Time.minute == 51 and (not self.invalidOrderTicket):
            self.invalidOrderTicket = self.MarketOnCloseOrder('SPY', 2)

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        MarketOnCloseOrder.SubmissionTimeBuffer = MarketOnCloseOrder.DefaultSubmissionTimeBuffer
        if self.validOrderTicket.Status != OrderStatus.Filled:
            raise Exception('Valid order failed to fill')
        if self.invalidOrderTicket.Status != OrderStatus.Invalid:
            raise Exception('Invalid order was not rejected')
        if self.validOrderTicketExtendedMarketHours.Status != OrderStatus.Filled:
            raise Exception('Valid order during extended market hours failed to fill')