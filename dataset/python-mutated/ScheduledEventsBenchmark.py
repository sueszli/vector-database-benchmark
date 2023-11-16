from AlgorithmImports import *

class ScheduledEventsBenchmark(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2011, 1, 1)
        self.SetEndDate(2022, 1, 1)
        self.SetCash(100000)
        self.AddEquity('SPY')
        for i in range(300):
            self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.AfterMarketOpen('SPY', i), self.Rebalance)
            self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.BeforeMarketClose('SPY', i), self.Rebalance)

    def OnData(self, data):
        if False:
            print('Hello World!')
        pass

    def Rebalance(self):
        if False:
            for i in range(10):
                print('nop')
        pass