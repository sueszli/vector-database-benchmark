from AlgorithmImports import *

class FutureOptionMultipleContractsInDifferentContractMonthsWithSameUnderlyingFutureRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.expectedSymbols = {self._createOption(datetime(2020, 3, 26), OptionRight.Call, 1650.0): False, self._createOption(datetime(2020, 3, 26), OptionRight.Put, 1540.0): False, self._createOption(datetime(2020, 2, 25), OptionRight.Call, 1600.0): False, self._createOption(datetime(2020, 2, 25), OptionRight.Put, 1545.0): False}
        self.UniverseSettings.ExtendedMarketHours = True
        self.SetStartDate(2020, 1, 4)
        self.SetEndDate(2020, 1, 6)
        goldFutures = self.AddFuture('GC', Resolution.Minute, Market.COMEX, extendedMarketHours=True)
        goldFutures.SetFilter(0, 365)
        self.AddFutureOption(goldFutures.Symbol)

    def OnData(self, data: Slice):
        if False:
            while True:
                i = 10
        for symbol in data.QuoteBars.Keys:
            if symbol in self.expectedSymbols and self.IsInRegularHours(symbol):
                invested = self.expectedSymbols[symbol]
                if not invested:
                    self.MarketOrder(symbol, 1)
                self.expectedSymbols[symbol] = True

    def OnEndOfAlgorithm(self):
        if False:
            i = 10
            return i + 15
        notEncountered = [str(k) for (k, v) in self.expectedSymbols.items() if not v]
        if any(notEncountered):
            raise AggregateException(f"Expected all Symbols encountered and invested in, but the following were not found: {', '.join(notEncountered)}")
        if not self.Portfolio.Invested:
            raise AggregateException('Expected holdings at the end of algorithm, but none were found.')

    def IsInRegularHours(self, symbol):
        if False:
            for i in range(10):
                print('nop')
        return self.Securities[symbol].Exchange.ExchangeOpen

    def _createOption(self, expiry: datetime, optionRight: OptionRight, strikePrice: float) -> Symbol:
        if False:
            return 10
        return Symbol.CreateOption(Symbol.CreateFuture('GC', Market.COMEX, datetime(2020, 4, 28)), Market.COMEX, OptionStyle.American, optionRight, strikePrice, expiry)