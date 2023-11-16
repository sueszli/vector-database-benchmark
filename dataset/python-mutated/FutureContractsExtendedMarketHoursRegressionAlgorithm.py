from AlgorithmImports import *

class FutureContractsExtendedMarketHoursRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2013, 10, 6)
        self.SetEndDate(2013, 10, 11)
        esFutureSymbol = Symbol.CreateFuture(Futures.Indices.SP500EMini, Market.CME, DateTime(2013, 12, 20))
        self._es = self.AddFutureContract(esFutureSymbol, Resolution.Hour, fillForward=True, extendedMarketHours=True)
        gcFutureSymbol = Symbol.CreateFuture(Futures.Metals.Gold, Market.COMEX, DateTime(2013, 10, 29))
        self._gc = self.AddFutureContract(gcFutureSymbol, Resolution.Hour, fillForward=True, extendedMarketHours=False)
        self._esRanOnRegularHours = False
        self._esRanOnExtendedHours = False
        self._gcRanOnRegularHours = False
        self._gcRanOnExtendedHours = False

    def OnData(self, slice):
        if False:
            for i in range(10):
                print('nop')
        sliceSymbols = set(slice.Keys)
        sliceSymbols.update(slice.Bars.Keys)
        sliceSymbols.update(slice.Ticks.Keys)
        sliceSymbols.update(slice.QuoteBars.Keys)
        sliceSymbols.update([x.Canonical for x in sliceSymbols])
        esIsInRegularHours = self._es.Exchange.Hours.IsOpen(self.Time, False)
        esIsInExtendedHours = not esIsInRegularHours and self._es.Exchange.Hours.IsOpen(self.Time, True)
        sliceHasESData = self._es.Symbol in sliceSymbols
        self._esRanOnRegularHours |= esIsInRegularHours and sliceHasESData
        self._esRanOnExtendedHours |= esIsInExtendedHours and sliceHasESData
        gcIsInRegularHours = self._gc.Exchange.Hours.IsOpen(self.Time, False)
        gcIsInExtendedHours = not gcIsInRegularHours and self._gc.Exchange.Hours.IsOpen(self.Time, True)
        sliceHasGCData = self._gc.Symbol in sliceSymbols
        self._gcRanOnRegularHours |= gcIsInRegularHours and sliceHasGCData
        self._gcRanOnExtendedHours |= gcIsInExtendedHours and sliceHasGCData

    def OnEndOfAlgorithm(self):
        if False:
            return 10
        if not self._esRanOnRegularHours:
            raise Exception(f'Algorithm should have run on regular hours for {self._es.Symbol} future, which enabled extended market hours')
        if not self._esRanOnExtendedHours:
            raise Exception(f'Algorithm should have run on extended hours for {self._es.Symbol} future, which enabled extended market hours')
        if not self._gcRanOnRegularHours:
            raise Exception(f'Algorithm should have run on regular hours for {self._gc.Symbol} future, which did not enable extended market hours')
        if self._gcRanOnExtendedHours:
            raise Exception(f'Algorithm should have not run on extended hours for {self._gc.Symbol} future, which did not enable extended market hours')