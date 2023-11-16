from AlgorithmImports import *
from RangeConsolidatorAlgorithm import RangeConsolidatorAlgorithm

class RangeConsolidatorWithTickAlgorithm(RangeConsolidatorAlgorithm):

    def GetRange(self):
        if False:
            print('Hello World!')
        return 5

    def GetResolution(self):
        if False:
            for i in range(10):
                print('nop')
        return Resolution.Tick

    def SetStartAndEndDates(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 7)