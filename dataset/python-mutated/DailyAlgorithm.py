from AlgorithmImports import *

class DailyAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 1, 1)
        self.SetEndDate(2014, 1, 1)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Daily)
        self.AddEquity('IBM', Resolution.Hour).SetLeverage(1.0)
        self.macd = self.MACD('SPY', 12, 26, 9, MovingAverageType.Wilders, Resolution.Daily, Field.Close)
        self.ema = self.EMA('IBM', 15 * 6, Resolution.Hour, Field.SevenBar)
        self.lastAction = None

    def OnData(self, data):
        if False:
            print('Hello World!')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.macd.IsReady:
            return
        if not data.ContainsKey('IBM'):
            return
        if data['IBM'] is None:
            self.Log('Price Missing Time: %s' % str(self.Time))
            return
        if self.lastAction is not None and self.lastAction.date() == self.Time.date():
            return
        self.lastAction = self.Time
        quantity = self.Portfolio['SPY'].Quantity
        if quantity <= 0 and self.macd.Current.Value > self.macd.Signal.Current.Value and (data['IBM'].Price > self.ema.Current.Value):
            self.SetHoldings('IBM', 0.25)
        elif quantity >= 0 and self.macd.Current.Value < self.macd.Signal.Current.Value and (data['IBM'].Price < self.ema.Current.Value):
            self.SetHoldings('IBM', -0.25)