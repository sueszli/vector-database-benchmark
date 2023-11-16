from AlgorithmImports import *

class HistoryRequestBenchmark(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2018, 1, 1)
        self.SetCash(10000)
        self.symbol = self.AddEquity('SPY').Symbol

    def OnEndOfDay(self, symbol):
        if False:
            print('Hello World!')
        minuteHistory = self.History([self.symbol], 60, Resolution.Minute)
        lastHourHigh = 0
        for (index, row) in minuteHistory.loc['SPY'].iterrows():
            if lastHourHigh < row['high']:
                lastHourHigh = row['high']
        dailyHistory = self.History([self.symbol], 1, Resolution.Daily).loc['SPY'].head()
        dailyHistoryHigh = dailyHistory['high']
        dailyHistoryLow = dailyHistory['low']
        dailyHistoryOpen = dailyHistory['open']