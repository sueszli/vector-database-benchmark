from datetime import timedelta
from AlgorithmImports import *

class TickHistoryRequestWithoutTickSubscriptionRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 8)
        spy = self.AddEquity('SPY', Resolution.Daily).Symbol
        ibm = self.AddEquity('IBM', Resolution.Hour).Symbol
        spyHistory = self.History[Tick](spy, timedelta(days=1), Resolution.Tick)
        if len(list(spyHistory)) == 0:
            raise Exception('SPY tick history is empty')
        ibmHistory = self.History[Tick](ibm, timedelta(days=1), Resolution.Tick)
        if len(list(ibmHistory)) == 0:
            raise Exception('IBM tick history is empty')
        spyIbmHistory = self.History[Tick]([spy, ibm], timedelta(days=1), Resolution.Tick)
        if len(list(spyIbmHistory)) == 0:
            raise Exception('Compound SPY and IBM tick history is empty')
        self.Quit()