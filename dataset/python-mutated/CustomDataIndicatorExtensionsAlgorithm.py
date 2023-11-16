from AlgorithmImports import *
from HistoryAlgorithm import *

class CustomDataIndicatorExtensionsAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2014, 1, 1)
        self.SetEndDate(2018, 1, 1)
        self.SetCash(25000)
        self.ibm = 'IBM'
        self.spy = 'SPY'
        self.AddData(CustomDataEquity, self.ibm, Resolution.Daily)
        self.AddData(CustomDataEquity, self.spy, Resolution.Daily)
        self.ibm_sma = self.SMA(self.ibm, 1, Resolution.Daily)
        self.spy_sma = self.SMA(self.spy, 1, Resolution.Daily)
        self.ratio = IndicatorExtensions.Over(self.spy_sma, self.ibm_sma)
        self.PlotIndicator('Ratio', self.ratio)
        self.PlotIndicator('Data', self.ibm_sma, self.spy_sma)

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        if not (self.ibm_sma.IsReady and self.spy_sma.IsReady and self.ratio.IsReady):
            return
        if not self.Portfolio.Invested and self.ratio.Current.Value > 1:
            self.MarketOrder(self.ibm, 100)
        elif self.ratio.Current.Value < 1:
            self.Liquidate()