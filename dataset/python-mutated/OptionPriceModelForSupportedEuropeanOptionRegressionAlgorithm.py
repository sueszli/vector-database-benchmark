from AlgorithmImports import *
from OptionPriceModelForOptionStylesBaseRegressionAlgorithm import OptionPriceModelForOptionStylesBaseRegressionAlgorithm

class OptionPriceModelForSupportedEuropeanOptionRegressionAlgorithm(OptionPriceModelForOptionStylesBaseRegressionAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2021, 1, 14)
        self.SetEndDate(2021, 1, 14)
        option = self.AddIndexOption('SPX', Resolution.Hour)
        option.PriceModel = OptionPriceModels.BlackScholes()
        self.SetWarmup(7, Resolution.Daily)
        self.Init(option, optionStyleIsSupported=True)