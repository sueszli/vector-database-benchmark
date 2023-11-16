from AlgorithmImports import *
from OptionPriceModelForUnsupportedAmericanOptionRegressionAlgorithm import OptionPriceModelForUnsupportedAmericanOptionRegressionAlgorithm

class OptionPriceModelForUnsupportedAmericanOptionTimeSpanWarmupRegressionAlgorithm(OptionPriceModelForUnsupportedAmericanOptionRegressionAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        OptionPriceModelForUnsupportedAmericanOptionRegressionAlgorithm.Initialize(self)
        self.SetWarmup(TimeSpan.FromDays(4))