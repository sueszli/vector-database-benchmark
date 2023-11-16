from AlgorithmImports import *

class ConvertToFrameworkAlgorithm(QCAlgorithm):
    """Demonstration algorithm showing how to easily convert an old algorithm into the framework."""
    FastEmaPeriod = 12
    SlowEmaPeriod = 26

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2004, 1, 1)
        self.SetEndDate(2015, 1, 1)
        self.symbol = self.AddSecurity(SecurityType.Equity, 'SPY', Resolution.Daily).Symbol
        self.macd = self.MACD(self.symbol, self.FastEmaPeriod, self.SlowEmaPeriod, 9, MovingAverageType.Exponential, Resolution.Daily)

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n        Args:\n            data: Slice object with your stock data'
        if not self.macd.IsReady or not data.ContainsKey(self.symbol) or data[self.symbol] is None:
            return
        holding = self.Portfolio[self.symbol]
        signalDeltaPercent = float(self.macd.Current.Value - self.macd.Signal.Current.Value) / float(self.macd.Fast.Current.Value)
        tolerance = 0.0025
        if holding.Quantity <= 0 and signalDeltaPercent > tolerance:
            self.EmitInsights(Insight.Price(self.symbol, timedelta(self.FastEmaPeriod), InsightDirection.Up))
            self.SetHoldings(self.symbol, 1)
        elif holding.Quantity >= 0 and signalDeltaPercent < -tolerance:
            self.EmitInsights(Insight.Price(self.symbol, timedelta(self.FastEmaPeriod), InsightDirection.Down))
            self.SetHoldings(self.symbol, -1)
        self.Plot('MACD', self.macd, self.macd.Signal)
        self.Plot(self.symbol.Value, self.macd.Fast, self.macd.Slow)
        self.Plot(self.symbol.Value, 'Open', data[self.symbol].Open)