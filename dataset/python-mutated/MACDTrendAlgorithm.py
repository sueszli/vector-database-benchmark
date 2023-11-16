from AlgorithmImports import *

class MACDTrendAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2004, 1, 1)
        self.SetEndDate(2015, 1, 1)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Daily)
        self.__macd = self.MACD('SPY', 12, 26, 9, MovingAverageType.Exponential, Resolution.Daily)
        self.__previous = datetime.min
        self.PlotIndicator('MACD', True, self.__macd, self.__macd.Signal)
        self.PlotIndicator('SPY', self.__macd.Fast, self.__macd.Slow)

    def OnData(self, data):
        if False:
            print('Hello World!')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if not self.__macd.IsReady:
            return
        if self.__previous.date() == self.Time.date():
            return
        tolerance = 0.0025
        holdings = self.Portfolio['SPY'].Quantity
        signalDeltaPercent = (self.__macd.Current.Value - self.__macd.Signal.Current.Value) / self.__macd.Fast.Current.Value
        if holdings <= 0 and signalDeltaPercent > tolerance:
            self.SetHoldings('SPY', 1.0)
        elif holdings >= 0 and signalDeltaPercent < -tolerance:
            self.Liquidate('SPY')
        self.__previous = self.Time