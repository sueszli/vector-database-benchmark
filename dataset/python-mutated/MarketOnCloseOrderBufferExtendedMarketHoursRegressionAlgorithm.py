from AlgorithmImports import *

class MarketOnCloseOrderBufferExtendedMarketHoursRegressionAlgorithm(QCAlgorithm):
    """This regression test is a version of "MarketOnCloseOrderBufferRegressionAlgorithm"
     where we test market-on-close modeling with data from the post market."""
    validOrderTicket = None
    invalidOrderTicket = None
    validOrderTicketExtendedMarketHours = None

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 8)
        self.AddEquity('SPY', Resolution.Minute, extendedMarketHours=True)

        def mocAtMidNight():
            if False:
                print('Hello World!')
            self.validOrderTicketAtMidnight = self.MarketOnCloseOrder('SPY', 2)
        self.Schedule.On(self.DateRules.Tomorrow, self.TimeRules.Midnight, mocAtMidNight)
        MarketOnCloseOrder.SubmissionTimeBuffer = timedelta(minutes=10)

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self.Time.hour == 15 and self.Time.minute == 49 and (not self.validOrderTicket):
            self.validOrderTicket = self.MarketOnCloseOrder('SPY', 2)
        if self.Time.hour == 15 and self.Time.minute == 51 and (not self.invalidOrderTicket):
            self.invalidOrderTicket = self.MarketOnCloseOrder('SPY', 2)
        if self.Time.hour == 16 and self.Time.minute == 48 and (not self.validOrderTicketExtendedMarketHours):
            self.validOrderTicketExtendedMarketHours = self.MarketOnCloseOrder('SPY', 2)

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        MarketOnCloseOrder.SubmissionTimeBuffer = MarketOnCloseOrder.DefaultSubmissionTimeBuffer
        if self.validOrderTicket.Status != OrderStatus.Filled:
            raise Exception('Valid order failed to fill')
        if self.invalidOrderTicket.Status != OrderStatus.Invalid:
            raise Exception('Invalid order was not rejected')
        if self.validOrderTicketExtendedMarketHours.Status != OrderStatus.Filled:
            raise Exception('Valid order during extended market hours failed to fill')