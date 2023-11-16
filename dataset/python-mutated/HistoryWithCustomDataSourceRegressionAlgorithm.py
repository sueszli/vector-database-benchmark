from AlgorithmImports import *

class HistoryWithCustomDataSourceRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2014, 6, 5)
        self.SetEndDate(2014, 6, 6)
        self.aapl = self.AddData(CustomData, 'AAPL', Resolution.Minute).Symbol
        self.spy = self.AddData(CustomData, 'SPY', Resolution.Minute).Symbol

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        aaplHistory = self.History(CustomData, self.aapl, self.StartDate, self.EndDate, Resolution.Minute, fillForward=False, extendedMarketHours=False, dataNormalizationMode=DataNormalizationMode.Raw).droplevel(0, axis=0)
        spyHistory = self.History(CustomData, self.spy, self.StartDate, self.EndDate, Resolution.Minute, fillForward=False, extendedMarketHours=False, dataNormalizationMode=DataNormalizationMode.Raw).droplevel(0, axis=0)
        if aaplHistory.size == 0 or spyHistory.size == 0:
            raise Exception('At least one of the history results is empty')
        if not aaplHistory.equals(spyHistory):
            raise Exception('Histories are not equal')

class CustomData(PythonData):
    """Custom data source for the regression test algorithm, which returns AAPL equity data regardless of the symbol requested."""

    def GetSource(self, config, date, isLiveMode):
        if False:
            for i in range(10):
                print('nop')
        return TradeBar().GetSource(SubscriptionDataConfig(config, CustomData, Symbol.Create('AAPL', SecurityType.Equity, config.Market)), date, isLiveMode)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            print('Hello World!')
        tradeBar = TradeBar.ParseEquity(config, line, date)
        data = CustomData()
        data.Time = tradeBar.Time
        data.Value = tradeBar.Value
        data.Close = tradeBar.Close
        data.Open = tradeBar.Open
        data.High = tradeBar.High
        data.Low = tradeBar.Low
        data.Volume = tradeBar.Volume
        return data