from AlgorithmImports import *
from RangeConsolidatorAlgorithm import RangeConsolidatorAlgorithm

class ClassicRangeConsolidatorAlgorithm(RangeConsolidatorAlgorithm):

    def CreateRangeConsolidator(self):
        if False:
            while True:
                i = 10
        return ClassicRangeConsolidator(self.GetRange())

    def OnDataConsolidated(self, sender, rangeBar):
        if False:
            while True:
                i = 10
        super().OnDataConsolidated(sender, rangeBar)
        if rangeBar.Volume == 0:
            raise Exception("All RangeBar's should have non-zero volume, but this doesn't")