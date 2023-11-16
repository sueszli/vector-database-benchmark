from AlgorithmImports import *

class LimitFillRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Second)

    def OnData(self, data):
        if False:
            print('Hello World!')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if data.ContainsKey('SPY'):
            if self.IsRoundHour(self.Time):
                negative = 1 if self.Time < self.StartDate + timedelta(days=2) else -1
                self.LimitOrder('SPY', negative * 10, data['SPY'].Price)

    def IsRoundHour(self, dateTime):
        if False:
            return 10
        'Verify whether datetime is round hour'
        return dateTime.minute == 0 and dateTime.second == 0

    def OnOrderEvent(self, orderEvent):
        if False:
            for i in range(10):
                print('nop')
        self.Debug(str(orderEvent))