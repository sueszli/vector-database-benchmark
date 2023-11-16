from AlgorithmImports import *
from OptionPriceModelForOptionStylesBaseRegressionAlgorithm import OptionPriceModelForOptionStylesBaseRegressionAlgorithm

class OptionPriceModelForUnsupportedEuropeanOptionRegressionAlgorithm(OptionPriceModelForOptionStylesBaseRegressionAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2021, 1, 14)
        self.SetEndDate(2021, 1, 14)
        option = self.AddIndexOption('SPX', Resolution.Hour)
        option.PriceModel = OptionPriceModels.BaroneAdesiWhaley()
        self.SetWarmup(7, Resolution.Daily)
        self.Init(option, optionStyleIsSupported=False)