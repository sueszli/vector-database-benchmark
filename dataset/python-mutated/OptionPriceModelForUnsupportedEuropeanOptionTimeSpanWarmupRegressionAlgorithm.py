from AlgorithmImports import *
from OptionPriceModelForUnsupportedEuropeanOptionRegressionAlgorithm import OptionPriceModelForUnsupportedEuropeanOptionRegressionAlgorithm

class OptionPriceModelForUnsupportedEuropeanOptionTimeSpanWarmupRegressionAlgorithm(OptionPriceModelForUnsupportedEuropeanOptionRegressionAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        OptionPriceModelForUnsupportedEuropeanOptionRegressionAlgorithm.Initialize(self)
        self.SetWarmup(TimeSpan.FromHours(24 * 9 + 23))