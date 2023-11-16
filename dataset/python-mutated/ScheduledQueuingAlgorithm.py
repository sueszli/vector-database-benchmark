from AlgorithmImports import *
from queue import Queue

class ScheduledQueuingAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2020, 9, 1)
        self.SetEndDate(2020, 9, 2)
        self.SetCash(100000)
        self.__numberOfSymbols = 2000
        self.__numberOfSymbolsFine = 1000
        self.SetUniverseSelection(FineFundamentalUniverseSelectionModel(self.CoarseSelectionFunction, self.FineSelectionFunction, None, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.queue = Queue()
        self.dequeue_size = 100
        self.AddEquity('SPY', Resolution.Minute)
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.At(0, 0), self.FillQueue)
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.Every(timedelta(minutes=60)), self.TakeFromQueue)

    def CoarseSelectionFunction(self, coarse):
        if False:
            i = 10
            return i + 15
        has_fundamentals = [security for security in coarse if security.HasFundamentalData]
        sorted_by_dollar_volume = sorted(has_fundamentals, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_dollar_volume[:self.__numberOfSymbols]]

    def FineSelectionFunction(self, fine):
        if False:
            while True:
                i = 10
        sorted_by_pe_ratio = sorted(fine, key=lambda x: x.ValuationRatios.PERatio, reverse=True)
        return [x.Symbol for x in sorted_by_pe_ratio[:self.__numberOfSymbolsFine]]

    def FillQueue(self):
        if False:
            print('Hello World!')
        securities = [security for security in self.ActiveSecurities.Values if security.Fundamentals is not None]
        self.queue.queue.clear()
        sorted_by_pe_ratio = sorted(securities, key=lambda x: x.Fundamentals.ValuationRatios.PERatio, reverse=True)
        for security in sorted_by_pe_ratio:
            self.queue.put(security.Symbol)

    def TakeFromQueue(self):
        if False:
            return 10
        symbols = [self.queue.get() for _ in range(min(self.dequeue_size, self.queue.qsize()))]
        self.History(symbols, 10, Resolution.Daily)
        self.Log(f'Symbols at {self.Time}: {[str(symbol) for symbol in symbols]}')