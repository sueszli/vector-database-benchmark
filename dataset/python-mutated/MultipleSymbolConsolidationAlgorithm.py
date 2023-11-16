from AlgorithmImports import *

class MultipleSymbolConsolidationAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        BarPeriod = TimeSpan.FromMinutes(10)
        SimpleMovingAveragePeriod = 10
        RollingWindowSize = 10
        self.Data = {}
        EquitySymbols = ['AAPL', 'SPY', 'IBM']
        ForexSymbols = ['EURUSD', 'USDJPY', 'EURGBP', 'EURCHF', 'USDCAD', 'USDCHF', 'AUDUSD', 'NZDUSD']
        self.SetStartDate(2014, 12, 1)
        self.SetEndDate(2015, 2, 1)
        for symbol in EquitySymbols:
            equity = self.AddEquity(symbol)
            self.Data[symbol] = SymbolData(equity.Symbol, BarPeriod, RollingWindowSize)
        for symbol in ForexSymbols:
            forex = self.AddForex(symbol)
            self.Data[symbol] = SymbolData(forex.Symbol, BarPeriod, RollingWindowSize)
        for (symbol, symbolData) in self.Data.items():
            symbolData.SMA = SimpleMovingAverage(self.CreateIndicatorName(symbol, 'SMA' + str(SimpleMovingAveragePeriod), Resolution.Minute), SimpleMovingAveragePeriod)
            consolidator = TradeBarConsolidator(BarPeriod) if symbolData.Symbol.SecurityType == SecurityType.Equity else QuoteBarConsolidator(BarPeriod)
            consolidator.DataConsolidated += self.OnDataConsolidated
            self.SubscriptionManager.AddConsolidator(symbolData.Symbol, consolidator)

    def OnDataConsolidated(self, sender, bar):
        if False:
            return 10
        self.Data[bar.Symbol.Value].SMA.Update(bar.Time, bar.Close)
        self.Data[bar.Symbol.Value].Bars.Add(bar)

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        for symbol in self.Data.keys():
            symbolData = self.Data[symbol]
            if symbolData.IsReady() and symbolData.WasJustUpdated(self.Time):
                if not self.Portfolio[symbol].Invested:
                    self.MarketOrder(symbol, 1)

    def OnEndOfDay(self, symbol):
        if False:
            for i in range(10):
                print('nop')
        i = 0
        for symbol in sorted(self.Data.keys()):
            symbolData = self.Data[symbol]
            i += 1
            if symbolData.IsReady() and i % 2 == 0:
                self.Plot(symbol, symbol, symbolData.SMA.Current.Value)

class SymbolData(object):

    def __init__(self, symbol, barPeriod, windowSize):
        if False:
            return 10
        self.Symbol = symbol
        self.BarPeriod = barPeriod
        self.Bars = RollingWindow[IBaseDataBar](windowSize)
        self.SMA = None

    def IsReady(self):
        if False:
            i = 10
            return i + 15
        return self.Bars.IsReady and self.SMA.IsReady

    def WasJustUpdated(self, current):
        if False:
            i = 10
            return i + 15
        return self.Bars.Count > 0 and self.Bars[0].Time == current - self.BarPeriod