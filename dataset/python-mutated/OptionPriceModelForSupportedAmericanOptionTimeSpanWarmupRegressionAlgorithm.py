from AlgorithmImports import *
from OptionPriceModelForSupportedAmericanOptionRegressionAlgorithm import OptionPriceModelForSupportedAmericanOptionRegressionAlgorithm

class OptionPriceModelForSupportedAmericanOptionTimeSpanWarmupRegressionAlgorithm(OptionPriceModelForSupportedAmericanOptionRegressionAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        OptionPriceModelForSupportedAmericanOptionRegressionAlgorithm.Initialize(self)
        self.SetWarmup(TimeSpan.FromDays(4))