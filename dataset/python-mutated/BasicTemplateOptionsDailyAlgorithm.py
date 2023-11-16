from AlgorithmImports import *

class BasicTemplateOptionsDailyAlgorithm(QCAlgorithm):
    UnderlyingTicker = 'GOOG'

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2015, 12, 23)
        self.SetEndDate(2016, 1, 20)
        self.SetCash(100000)
        self.optionExpired = False
        equity = self.AddEquity(self.UnderlyingTicker, Resolution.Daily)
        option = self.AddOption(self.UnderlyingTicker, Resolution.Daily)
        self.option_symbol = option.Symbol
        option.SetFilter(lambda u: u.CallsOnly().Strikes(0, 1).Expiration(0, 30))
        self.SetBenchmark(equity.Symbol)

    def OnData(self, slice):
        if False:
            while True:
                i = 10
        if self.Portfolio.Invested:
            return
        chain = slice.OptionChains.GetValue(self.option_symbol)
        if chain is None:
            return
        contracts = sorted(chain, key=lambda x: x.Expiry)
        if len(contracts) == 0:
            return
        symbol = contracts[0].Symbol
        self.MarketOrder(symbol, 1)

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        self.Log(str(orderEvent))
        if 'OTM' in orderEvent.Message:
            if orderEvent.UtcTime.month != 1 and orderEvent.UtcTime.day != 16 and (orderEvent.UtcTime.hour != 5):
                raise AssertionError(f'Expiry event was not at the correct time, {orderEvent.UtcTime}')
            self.optionExpired = True

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.optionExpired:
            raise AssertionError('Algorithm did not process the option expiration like expected')