from AlgorithmImports import *

class RangeConsolidatorAlgorithm(QCAlgorithm):

    def GetResolution(self):
        if False:
            while True:
                i = 10
        return Resolution.Daily

    def GetRange(self):
        if False:
            i = 10
            return i + 15
        return 100

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartAndEndDates()
        self.AddEquity('SPY', self.GetResolution())
        rangeConsolidator = self.CreateRangeConsolidator()
        rangeConsolidator.DataConsolidated += self.OnDataConsolidated
        self.firstDataConsolidated = None
        self.SubscriptionManager.AddConsolidator('SPY', rangeConsolidator)

    def SetStartAndEndDates(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        if self.firstDataConsolidated == None:
            raise Exception('The consolidator should have consolidated at least one RangeBar, but it did not consolidated any one')

    def CreateRangeConsolidator(self):
        if False:
            return 10
        return RangeConsolidator(self.GetRange())

    def OnDataConsolidated(self, sender, rangeBar):
        if False:
            return 10
        if self.firstDataConsolidated is None:
            self.firstDataConsolidated = rangeBar
        if round(rangeBar.High - rangeBar.Low, 2) != self.GetRange() * 0.01:
            raise Exception(f"The difference between the High and Low for all RangeBar's should be {self.GetRange() * 0.01}, but for this RangeBar was {round(rangeBar.Low - rangeBar.High, 2)}")