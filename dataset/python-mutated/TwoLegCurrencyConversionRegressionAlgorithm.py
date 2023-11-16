from AlgorithmImports import *

class TwoLegCurrencyConversionRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2018, 4, 4)
        self.SetEndDate(2018, 4, 4)
        self.SetBrokerageModel(BrokerageName.GDAX, AccountType.Cash)
        self.SetAccountCurrency('ETH')
        self.SetCash('ETH', 100000)
        self.SetCash('LTC', 100000)
        self.SetCash('USD', 100000)
        self._ethUsdSymbol = self.AddCrypto('ETHUSD', Resolution.Minute).Symbol
        self._ltcUsdSymbol = self.AddCrypto('LTCUSD', Resolution.Minute).Symbol

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        if not self.Portfolio.Invested:
            self.MarketOrder(self._ltcUsdSymbol, 1)

    def OnEndOfAlgorithm(self):
        if False:
            i = 10
            return i + 15
        ltcCash = self.Portfolio.CashBook['LTC']
        conversionSymbols = [x.Symbol for x in ltcCash.CurrencyConversion.ConversionRateSecurities]
        if len(conversionSymbols) != 2:
            raise ValueError(f'Expected two conversion rate securities for LTC to ETH, is {len(conversionSymbols)}')
        if conversionSymbols[0] != self._ltcUsdSymbol:
            raise ValueError(f'Expected first conversion rate security from LTC to ETH to be {self._ltcUsdSymbol}, is {conversionSymbols[0]}')
        if conversionSymbols[1] != self._ethUsdSymbol:
            raise ValueError(f'Expected second conversion rate security from LTC to ETH to be {self._ethUsdSymbol}, is {conversionSymbols[1]}')
        ltcUsdValue = self.Securities[self._ltcUsdSymbol].GetLastData().Value
        ethUsdValue = self.Securities[self._ethUsdSymbol].GetLastData().Value
        expectedConversionRate = ltcUsdValue / ethUsdValue
        actualConversionRate = ltcCash.ConversionRate
        if actualConversionRate != expectedConversionRate:
            raise ValueError(f'Expected conversion rate from LTC to ETH to be {expectedConversionRate}, is {actualConversionRate}')