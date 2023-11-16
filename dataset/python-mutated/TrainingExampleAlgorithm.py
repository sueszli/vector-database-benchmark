from AlgorithmImports import *
from time import sleep

class TrainingExampleAlgorithm(QCAlgorithm):
    """Example algorithm showing how to use QCAlgorithm.Train method"""

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 14)
        self.AddEquity('SPY', Resolution.Daily)
        self.Train(self.TrainingMethod)
        self.Train(self.DateRules.Every(DayOfWeek.Sunday), self.TimeRules.At(8, 0), self.TrainingMethod)

    def TrainingMethod(self):
        if False:
            while True:
                i = 10
        self.Log(f'Start training at {self.Time}')
        history = self.History(['SPY'], 200, Resolution.Daily)
        pass