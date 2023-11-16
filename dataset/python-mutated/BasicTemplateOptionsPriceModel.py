from AlgorithmImports import *

class BasicTemplateOptionsPriceModel(QCAlgorithm):
    """Example demonstrating how to define an option price model."""

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2020, 1, 5)
        self.SetCash(100000)
        option = self.AddOption('AAPL')
        self.optionSymbol = option.Symbol
        option.SetFilter(-3, +3, 0, 31)
        option.PriceModel = OptionPriceModels.CrankNicolsonFD()
        self.SetWarmUp(30, Resolution.Daily)

    def OnData(self, slice):
        if False:
            i = 10
            return i + 15
        'OnData will test whether the option contracts has a non-zero Greeks.Delta'
        if self.IsWarmingUp or not slice.OptionChains.ContainsKey(self.optionSymbol):
            return
        chain = slice.OptionChains[self.optionSymbol]
        if not any([x for x in chain if x.Greeks.Delta != 0]):
            self.Log(f'No contract with Delta != 0')