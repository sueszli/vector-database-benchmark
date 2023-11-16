from AlgorithmImports import *

class IndicatorWarmupAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(1000000)
        self.AddEquity('SPY')
        self.AddEquity('IBM')
        self.AddEquity('BAC')
        self.AddEquity('GOOG', Resolution.Daily)
        self.AddEquity('GOOGL', Resolution.Daily)
        self.__sd = {}
        for security in self.Securities:
            self.__sd[security.Key] = self.SymbolData(security.Key, self)
        self.SetWarmup(self.SymbolData.RequiredBarsWarmup)

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if self.IsWarmingUp:
            return
        for sd in self.__sd.values():
            lastPriceTime = sd.Close.Current.Time
            if self.RoundDown(lastPriceTime, sd.Security.SubscriptionDataConfig.Increment):
                sd.Update()

    def OnOrderEvent(self, fill):
        if False:
            while True:
                i = 10
        sd = self.__sd.get(fill.Symbol, None)
        if sd is not None:
            sd.OnOrderEvent(fill)

    def RoundDown(self, time, increment):
        if False:
            return 10
        if increment.days != 0:
            return time.hour == 0 and time.minute == 0 and (time.second == 0)
        else:
            return time.second == 0

    class SymbolData:
        RequiredBarsWarmup = 40
        PercentTolerance = 0.001
        PercentGlobalStopLoss = 0.01
        LotSize = 10

        def __init__(self, symbol, algorithm):
            if False:
                print('Hello World!')
            self.Symbol = symbol
            self.__algorithm = algorithm
            self.__currentStopLoss = None
            self.Security = algorithm.Securities[symbol]
            self.Close = algorithm.Identity(symbol)
            self.ADX = algorithm.ADX(symbol, 14)
            self.EMA = algorithm.EMA(symbol, 14)
            self.MACD = algorithm.MACD(symbol, 12, 26, 9)
            self.IsReady = self.Close.IsReady and self.ADX.IsReady and self.EMA.IsReady and self.MACD.IsReady
            self.IsUptrend = False
            self.IsDowntrend = False

        def Update(self):
            if False:
                i = 10
                return i + 15
            self.IsReady = self.Close.IsReady and self.ADX.IsReady and self.EMA.IsReady and self.MACD.IsReady
            tolerance = 1 - self.PercentTolerance
            self.IsUptrend = self.MACD.Signal.Current.Value > self.MACD.Current.Value * tolerance and self.EMA.Current.Value > self.Close.Current.Value * tolerance
            self.IsDowntrend = self.MACD.Signal.Current.Value < self.MACD.Current.Value * tolerance and self.EMA.Current.Value < self.Close.Current.Value * tolerance
            self.TryEnter()
            self.TryExit()

        def TryEnter(self):
            if False:
                for i in range(10):
                    print('nop')
            if self.Security.Invested:
                return False
            qty = 0
            limit = 0.0
            if self.IsUptrend:
                qty = self.LotSize
                limit = self.Security.Low
            elif self.IsDowntrend:
                qty = -self.LotSize
                limit = self.Security.High
            if qty != 0:
                ticket = self.__algorithm.LimitOrder(self.Symbol, qty, limit, 'TryEnter at: {0}'.format(limit))

        def TryExit(self):
            if False:
                return 10
            if not self.Security.Invested:
                return
            limit = 0
            qty = self.Security.Holdings.Quantity
            exitTolerance = 1 + 2 * self.PercentTolerance
            if self.Security.Holdings.IsLong and self.Close.Current.Value * exitTolerance < self.EMA.Current.Value:
                limit = self.Security.High
            elif self.Security.Holdings.IsShort and self.Close.Current.Value > self.EMA.Current.Value * exitTolerance:
                limit = self.Security.Low
            if limit != 0:
                ticket = self.__algorithm.LimitOrder(self.Symbol, -qty, limit, 'TryExit at: {0}'.format(limit))

        def OnOrderEvent(self, fill):
            if False:
                while True:
                    i = 10
            if fill.Status != OrderStatus.Filled:
                return
            qty = self.Security.Holdings.Quantity
            if self.Security.Invested:
                stop = fill.FillPrice * (1 - self.PercentGlobalStopLoss) if self.Security.Holdings.IsLong else fill.FillPrice * (1 + self.PercentGlobalStopLoss)
                self.__currentStopLoss = self.__algorithm.StopMarketOrder(self.Symbol, -qty, stop, 'StopLoss at: {0}'.format(stop))
            elif self.__currentStopLoss is not None and self.__currentStopLoss.Status is not OrderStatus.Filled:
                self.__currentStopLoss.Cancel('Exited position')
                self.__currentStopLoss = None