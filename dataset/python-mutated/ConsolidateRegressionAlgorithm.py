from AlgorithmImports import *
from CustomDataRegressionAlgorithm import Bitcoin

class ConsolidateRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 20)
        SP500 = Symbol.Create(Futures.Indices.SP500EMini, SecurityType.Future, Market.CME)
        self._symbol = _symbol = self.FutureChainProvider.GetFutureContractList(SP500, self.StartDate)[0]
        self.AddFutureContract(_symbol)
        self._consolidationCounts = [0] * 6
        self._smas = [SimpleMovingAverage(10) for x in self._consolidationCounts]
        self._lastSmaUpdates = [datetime.min for x in self._consolidationCounts]
        self._monthlyConsolidatorSma = SimpleMovingAverage(10)
        self._monthlyConsolidationCount = 0
        self._weeklyConsolidatorSma = SimpleMovingAverage(10)
        self._weeklyConsolidationCount = 0
        self._lastWeeklySmaUpdate = datetime.min
        self.Consolidate(_symbol, Calendar.Monthly, lambda bar: self.UpdateMonthlyConsolidator(bar, -1))
        self.Consolidate(_symbol, Calendar.Weekly, TickType.Trade, lambda bar: self.UpdateWeeklyConsolidator(bar))
        self.Consolidate(_symbol, Resolution.Daily, lambda bar: self.UpdateTradeBar(bar, 0))
        self.Consolidate(_symbol, Resolution.Daily, TickType.Quote, lambda bar: self.UpdateQuoteBar(bar, 1))
        self.Consolidate(_symbol, timedelta(1), lambda bar: self.UpdateTradeBar(bar, 2))
        self.Consolidate(_symbol, timedelta(1), TickType.Quote, lambda bar: self.UpdateQuoteBar(bar, 3))
        self.Consolidate(_symbol, timedelta(1), None, lambda bar: self.UpdateTradeBar(bar, 4))
        self.Consolidate(_symbol, Resolution.Daily, None, lambda bar: self.UpdateTradeBar(bar, 5))
        self._customDataConsolidator = 0
        customSymbol = self.AddData(Bitcoin, 'BTC', Resolution.Minute).Symbol
        self.Consolidate(customSymbol, timedelta(1), lambda bar: self.IncrementCounter(1))
        self._customDataConsolidator2 = 0
        self.Consolidate(customSymbol, Resolution.Daily, lambda bar: self.IncrementCounter(2))

    def IncrementCounter(self, id):
        if False:
            while True:
                i = 10
        if id == 1:
            self._customDataConsolidator += 1
        if id == 2:
            self._customDataConsolidator2 += 1

    def UpdateTradeBar(self, bar, position):
        if False:
            print('Hello World!')
        self._smas[position].Update(bar.EndTime, bar.Volume)
        self._lastSmaUpdates[position] = bar.EndTime
        self._consolidationCounts[position] += 1

    def UpdateQuoteBar(self, bar, position):
        if False:
            i = 10
            return i + 15
        self._smas[position].Update(bar.EndTime, bar.Ask.High)
        self._lastSmaUpdates[position] = bar.EndTime
        self._consolidationCounts[position] += 1

    def UpdateMonthlyConsolidator(self, bar):
        if False:
            print('Hello World!')
        self._monthlyConsolidatorSma.Update(bar.EndTime, bar.Volume)
        self._monthlyConsolidationCount += 1

    def UpdateWeeklyConsolidator(self, bar):
        if False:
            while True:
                i = 10
        self._weeklyConsolidatorSma.Update(bar.EndTime, bar.Volume)
        self._lastWeeklySmaUpdate = bar.EndTime
        self._weeklyConsolidationCount += 1

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        expectedConsolidations = 8
        expectedWeeklyConsolidations = 1
        if any((i != expectedConsolidations for i in self._consolidationCounts)) or self._weeklyConsolidationCount != expectedWeeklyConsolidations or self._customDataConsolidator == 0 or (self._customDataConsolidator2 == 0):
            raise ValueError('Unexpected consolidation count')
        for (i, sma) in enumerate(self._smas):
            if sma.Samples != expectedConsolidations:
                raise Exception(f'Expected {expectedConsolidations} samples in each SMA but found {sma.Samples} in SMA in index {i}')
            lastUpdate = self._lastSmaUpdates[i]
            if sma.Current.Time != lastUpdate:
                raise Exception(f'Expected SMA in index {i} to have been last updated at {lastUpdate} but was {sma.Current.Time}')
        if self._monthlyConsolidationCount != 0 or self._monthlyConsolidatorSma.Samples != 0:
            raise Exception('Expected monthly consolidator to not have consolidated any data')
        if self._weeklyConsolidatorSma.Samples != expectedWeeklyConsolidations:
            raise Exception(f'Expected {expectedWeeklyConsolidations} samples in the weekly consolidator SMA but found {self._weeklyConsolidatorSma.Samples}')
        if self._weeklyConsolidatorSma.Current.Time != self._lastWeeklySmaUpdate:
            raise Exception(f'Expected weekly consolidator SMA to have been last updated at {self._lastWeeklySmaUpdate} but was {self._weeklyConsolidatorSma.Current.Time}')

    def OnData(self, data):
        if False:
            print('Hello World!')
        if not self.Portfolio.Invested:
            self.SetHoldings(self._symbol, 0.5)