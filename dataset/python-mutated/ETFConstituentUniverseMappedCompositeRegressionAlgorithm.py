from AlgorithmImports import *

class ETFConstituentUniverseFilterFunctionRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2011, 2, 1)
        self.SetEndDate(2011, 4, 4)
        self.SetCash(100000)
        self.filterDateConstituentSymbolCount = {}
        self.constituentDataEncountered = {}
        self.constituentSymbols = []
        self.mappingEventOccurred = False
        self.UniverseSettings.Resolution = Resolution.Hour
        self.aapl = Symbol.Create('AAPL', SecurityType.Equity, Market.USA)
        self.qqq = self.AddEquity('QQQ', Resolution.Daily).Symbol
        self.AddUniverse(self.Universe.ETF(self.qqq, self.UniverseSettings, self.FilterETFs))

    def FilterETFs(self, constituents):
        if False:
            print('Hello World!')
        constituentSymbols = [i.Symbol for i in constituents]
        if self.aapl not in constituentSymbols:
            raise Exception('AAPL not found in QQQ constituents')
        self.filterDateConstituentSymbolCount[self.UtcTime.date()] = len(constituentSymbols)
        for symbol in constituentSymbols:
            self.constituentSymbols.append(symbol)
        self.constituentSymbols = list(set(self.constituentSymbols))
        return constituentSymbols

    def OnData(self, data):
        if False:
            while True:
                i = 10
        if len(data.SymbolChangedEvents) != 0:
            for symbolChanged in data.SymbolChangedEvents.Values:
                if symbolChanged.Symbol != self.qqq:
                    raise Exception(f'Mapped symbol is not QQQ. Instead, found: {symbolChanged.Symbol}')
                if symbolChanged.OldSymbol != 'QQQQ':
                    raise Exception(f'Old QQQ Symbol is not QQQQ. Instead, found: {symbolChanged.OldSymbol}')
                if symbolChanged.NewSymbol != 'QQQ':
                    raise Exception(f'New QQQ Symbol is not QQQ. Instead, found: {symbolChanged.NewSymbol}')
                self.mappingEventOccurred = True
        if self.qqq in data and len([i for i in data.Keys]) == 1:
            return
        if self.UtcTime.date() not in self.constituentDataEncountered:
            self.constituentDataEncountered[self.UtcTime.date()] = False
        if len([i for i in data.Keys if i in self.constituentSymbols]) != 0:
            self.constituentDataEncountered[self.UtcTime.date()] = True
        if not self.Portfolio.Invested:
            self.SetHoldings(self.aapl, 0.5)

    def OnEndOfAlgorithm(self):
        if False:
            i = 10
            return i + 15
        if len(self.filterDateConstituentSymbolCount) != 2:
            raise Exception(f'ETF constituent filtering function was not called 2 times (actual: {len(self.filterDateConstituentSymbolCount)}')
        if not self.mappingEventOccurred:
            raise Exception('No mapping/SymbolChangedEvent occurred. Expected for QQQ to be mapped from QQQQ -> QQQ')
        for (constituentDate, constituentsCount) in self.filterDateConstituentSymbolCount.items():
            if constituentsCount < 25:
                raise Exception(f'Expected 25 or more constituents in filter function on {constituentDate}, found {constituentsCount}')
        for (constituentDate, constituentEncountered) in self.constituentDataEncountered.items():
            if not constituentEncountered:
                raise Exception(f"Received data in OnData(...) but it did not contain any constituent data on {constituentDate.strftime('%Y-%m-%d %H:%M:%S.%f')}")