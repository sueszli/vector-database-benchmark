from AlgorithmImports import *
from OptionPriceModelForOptionStylesBaseRegressionAlgorithm import OptionPriceModelForOptionStylesBaseRegressionAlgorithm

class OptionPriceModelForSupportedAmericanOptionRegressionAlgorithm(OptionPriceModelForOptionStylesBaseRegressionAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2014, 6, 9)
        self.SetEndDate(2014, 6, 9)
        option = self.AddOption('AAPL', Resolution.Minute)
        option.PriceModel = OptionPriceModels.BaroneAdesiWhaley()
        self.SetWarmup(2, Resolution.Daily)
        self.Init(option, optionStyleIsSupported=True)