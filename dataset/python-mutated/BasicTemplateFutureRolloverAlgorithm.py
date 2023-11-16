from AlgorithmImports import *

class BasicTemplateFutureRolloverAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 12, 10)
        self.SetCash(1000000)
        self.symbol_data_by_symbol = {}
        futures = [Futures.Indices.SP500EMini]
        for future in futures:
            continuous_contract = self.AddFuture(future, resolution=Resolution.Daily, extendedMarketHours=True, dataNormalizationMode=DataNormalizationMode.BackwardsRatio, dataMappingMode=DataMappingMode.OpenInterest, contractDepthOffset=0)
            symbol_data = SymbolData(self, continuous_contract)
            self.symbol_data_by_symbol[continuous_contract.Symbol] = symbol_data

    def OnData(self, slice):
        if False:
            while True:
                i = 10
        for (symbol, symbol_data) in self.symbol_data_by_symbol.items():
            symbol_data.Update(slice)
            if not symbol_data.IsReady or not slice.Bars.ContainsKey(symbol):
                return
            ema_current_value = symbol_data.EMA.Current.Value
            if ema_current_value < symbol_data.Price and (not symbol_data.IsLong):
                self.MarketOrder(symbol_data.Mapped, 1)
            elif ema_current_value > symbol_data.Price and (not symbol_data.IsShort):
                self.MarketOrder(symbol_data.Mapped, -1)

class SymbolData:

    def __init__(self, algorithm, future):
        if False:
            i = 10
            return i + 15
        self._algorithm = algorithm
        self._future = future
        self.EMA = algorithm.EMA(future.Symbol, 20, Resolution.Daily)
        self.Price = 0
        self.IsLong = False
        self.IsShort = False
        self.Reset()

    @property
    def Symbol(self):
        if False:
            i = 10
            return i + 15
        return self._future.Symbol

    @property
    def Mapped(self):
        if False:
            print('Hello World!')
        return self._future.Mapped

    @property
    def IsReady(self):
        if False:
            while True:
                i = 10
        return self.Mapped is not None and self.EMA.IsReady

    def Update(self, slice):
        if False:
            print('Hello World!')
        if slice.SymbolChangedEvents.ContainsKey(self.Symbol):
            changed_event = slice.SymbolChangedEvents[self.Symbol]
            old_symbol = changed_event.OldSymbol
            new_symbol = changed_event.NewSymbol
            tag = f'Rollover - Symbol changed at {self._algorithm.Time}: {old_symbol} -> {new_symbol}'
            quantity = self._algorithm.Portfolio[old_symbol].Quantity
            self._algorithm.Liquidate(old_symbol, tag=tag)
            self._algorithm.MarketOrder(new_symbol, quantity, tag=tag)
            self.Reset()
        self.Price = slice.Bars[self.Symbol].Price if slice.Bars.ContainsKey(self.Symbol) else self.Price
        self.IsLong = self._algorithm.Portfolio[self.Mapped].IsLong
        self.IsShort = self._algorithm.Portfolio[self.Mapped].IsShort

    def Reset(self):
        if False:
            return 10
        self.EMA.Reset()
        self._algorithm.WarmUpIndicator(self.Symbol, self.EMA, Resolution.Daily)

    def Dispose(self):
        if False:
            for i in range(10):
                print('nop')
        self.EMA.Reset()