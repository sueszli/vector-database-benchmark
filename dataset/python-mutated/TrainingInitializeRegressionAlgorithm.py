from AlgorithmImports import *
from time import sleep

class TrainingInitializeRegressionAlgorithm(QCAlgorithm):
    """Example algorithm showing how to use QCAlgorithm.Train method"""

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.AddEquity('SPY', Resolution.Daily)
        self.Train(lambda : sleep(150))
        self.Train(self.DateRules.Tomorrow, self.TimeRules.Midnight, lambda : sleep(60))