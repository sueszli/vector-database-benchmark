from AlgorithmImports import *

class OptionPriceModelForOptionStylesBaseRegressionAlgorithm(QCAlgorithm):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._optionStyleIsSupported = False
        self._checkGreeks = True
        self._triedGreeksCalculation = False
        self._option = None

    def OnData(self, slice):
        if False:
            for i in range(10):
                print('nop')
        if self.IsWarmingUp:
            return
        for kvp in slice.OptionChains:
            if self._option is None or kvp.Key != self._option.Symbol:
                continue
            self.CheckGreeks([contract for contract in kvp.Value])

    def OnEndOfDay(self, symbol):
        if False:
            print('Hello World!')
        self._checkGreeks = True

    def OnEndOfAlgorithm(self):
        if False:
            return 10
        if not self._triedGreeksCalculation:
            raise Exception('Expected greeks to be accessed')

    def Init(self, option, optionStyleIsSupported):
        if False:
            for i in range(10):
                print('nop')
        self._option = option
        self._optionStyleIsSupported = optionStyleIsSupported
        self._checkGreeks = True
        self._triedGreeksCalculation = False

    def CheckGreeks(self, contracts):
        if False:
            return 10
        if not self._checkGreeks or len(contracts) == 0:
            return
        self._checkGreeks = False
        self._triedGreeksCalculation = True
        for contract in contracts:
            greeks = Greeks()
            try:
                greeks = contract.Greeks
                optionStyleStr = 'American' if self._option.Style == OptionStyle.American else 'European'
                if not self._optionStyleIsSupported:
                    raise Exception(f'Expected greeks not to be calculated for {contract.Symbol.Value}, an {optionStyleStr} style option, using {type(self._option.PriceModel).__name__}, which does not support them, but they were')
            except ArgumentException:
                if self._optionStyleIsSupported:
                    raise Exception(f'Expected greeks to be calculated for {contract.Symbol.Value}, an {optionStyleStr} style option, using {type(self._option.PriceModel).__name__}, which supports them, but they were not')
            if self._optionStyleIsSupported and (contract.Right == OptionRight.Call and (greeks.Delta < 0.0 or greeks.Delta > 1.0 or greeks.Rho < 0.0) or (contract.Right == OptionRight.Put and (greeks.Delta < -1.0 or greeks.Delta > 0.0 or greeks.Rho > 0.0)) or greeks.Theta == 0.0 or (greeks.Vega < 0.0) or (greeks.Gamma < 0.0)):
                raise Exception(f'Expected greeks to have valid values. Greeks were: Delta: {greeks.Delta}, Rho: {greeks.Rho}, Theta: {greeks.Theta}, Vega: {greeks.Vega}, Gamma: {greeks.Gamma}')