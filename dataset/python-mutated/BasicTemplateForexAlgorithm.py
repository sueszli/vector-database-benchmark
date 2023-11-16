from AlgorithmImports import *

class BasicTemplateForexAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetCash(100000)
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.AddForex('EURUSD', Resolution.Minute)
        self.AddForex('GBPUSD', Resolution.Minute)
        self.AddForex('EURGBP', Resolution.Minute)
        self.History(5, Resolution.Daily)
        self.History(5, Resolution.Hour)
        self.History(5, Resolution.Minute)
        history = self.History(TimeSpan.FromSeconds(5), Resolution.Second)
        for data in sorted(history, key=lambda x: x.Time):
            for key in data.Keys:
                self.Log(str(key.Value) + ': ' + str(data.Time) + ' > ' + str(data[key].Value))

    def OnData(self, data):
        if False:
            while True:
                i = 10
        for key in data.Keys:
            self.Log(str(key.Value) + ': ' + str(data.Time) + ' > ' + str(data[key].Value))