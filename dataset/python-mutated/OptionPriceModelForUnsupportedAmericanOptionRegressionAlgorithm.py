from AlgorithmImports import *
from OptionPriceModelForOptionStylesBaseRegressionAlgorithm import OptionPriceModelForOptionStylesBaseRegressionAlgorithm

class OptionPriceModelForUnsupportedAmericanOptionRegressionAlgorithm(OptionPriceModelForOptionStylesBaseRegressionAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2014, 6, 9)
        self.SetEndDate(2014, 6, 9)
        option = self.AddOption('AAPL', Resolution.Minute)
        option.PriceModel = OptionPriceModels.BlackScholes()
        self.SetWarmup(2, Resolution.Daily)
        self.Init(option, optionStyleIsSupported=False)