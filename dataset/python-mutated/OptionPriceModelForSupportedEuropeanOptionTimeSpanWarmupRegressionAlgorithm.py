from AlgorithmImports import *
from OptionPriceModelForSupportedEuropeanOptionRegressionAlgorithm import OptionPriceModelForSupportedEuropeanOptionRegressionAlgorithm

class OptionPriceModelForSupportedEuropeanOptionTimeSpanWarmupRegressionAlgorithm(OptionPriceModelForSupportedEuropeanOptionRegressionAlgorithm):

    def Initialize(self):
        if False:
            return 10
        OptionPriceModelForSupportedEuropeanOptionRegressionAlgorithm.Initialize(self)
        self.SetWarmup(TimeSpan.FromHours(24 * 9 + 23))