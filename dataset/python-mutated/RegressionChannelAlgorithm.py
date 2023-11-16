from AlgorithmImports import *

class RegressionChannelAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetCash(100000)
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2015, 1, 1)
        equity = self.AddEquity('SPY', Resolution.Minute)
        self._spy = equity.Symbol
        self._holdings = equity.Holdings
        self._rc = self.RC(self._spy, 30, 2, Resolution.Daily)
        stockPlot = Chart('Trade Plot')
        stockPlot.AddSeries(Series('Buy', SeriesType.Scatter, 0))
        stockPlot.AddSeries(Series('Sell', SeriesType.Scatter, 0))
        stockPlot.AddSeries(Series('UpperChannel', SeriesType.Line, 0))
        stockPlot.AddSeries(Series('LowerChannel', SeriesType.Line, 0))
        stockPlot.AddSeries(Series('Regression', SeriesType.Line, 0))
        self.AddChart(stockPlot)

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        if not self._rc.IsReady or not data.ContainsKey(self._spy):
            return
        if data[self._spy] is None:
            return
        value = data[self._spy].Value
        if self._holdings.Quantity <= 0 and value < self._rc.LowerChannel.Current.Value:
            self.SetHoldings(self._spy, 1)
            self.Plot('Trade Plot', 'Buy', value)
        if self._holdings.Quantity >= 0 and value > self._rc.UpperChannel.Current.Value:
            self.SetHoldings(self._spy, -1)
            self.Plot('Trade Plot', 'Sell', value)

    def OnEndOfDay(self, symbol):
        if False:
            while True:
                i = 10
        self.Plot('Trade Plot', 'UpperChannel', self._rc.UpperChannel.Current.Value)
        self.Plot('Trade Plot', 'LowerChannel', self._rc.LowerChannel.Current.Value)
        self.Plot('Trade Plot', 'Regression', self._rc.LinearRegression.Current.Value)