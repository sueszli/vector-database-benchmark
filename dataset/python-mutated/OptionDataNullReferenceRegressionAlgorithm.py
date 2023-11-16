from AlgorithmImports import *

class OptionDataNullReferenceRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2016, 12, 1)
        self.SetEndDate(2017, 1, 1)
        self.SetCash(500000)
        self.AddEquity('DUST')
        option = self.AddOption('DUST')
        option.SetFilter(self.UniverseFunc)

    def UniverseFunc(self, universe):
        if False:
            i = 10
            return i + 15
        return universe.IncludeWeeklys().Strikes(-1, +1).Expiration(timedelta(25), timedelta(100))