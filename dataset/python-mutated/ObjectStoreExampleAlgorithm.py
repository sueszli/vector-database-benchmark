from AlgorithmImports import *
from io import StringIO

class ObjectStoreExampleAlgorithm(QCAlgorithm):
    """This algorithm showcases some features of the IObjectStore feature.
    One use case is to make consecutive backtests run faster by caching the results of
    potentially time consuming operations. In this example, we save the results of a
    history call. This pattern can be equally applied to a machine learning model being
    trained and then saving the model weights in the object store.
    """
    SPY_Close_ObjectStore_Key = 'spy_close'
    SPY_Close_History = RollingWindow[IndicatorDataPoint](252)
    SPY_Close_EMA10_History = RollingWindow[IndicatorDataPoint](252)
    SPY_Close_EMA50_History = RollingWindow[IndicatorDataPoint](252)

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SPY = self.AddEquity('SPY', Resolution.Minute).Symbol
        self.SPY_Close = self.Identity(self.SPY, Resolution.Daily)
        self.SPY_Close_EMA10 = IndicatorExtensions.EMA(self.SPY_Close, 10)
        self.SPY_Close_EMA50 = IndicatorExtensions.EMA(self.SPY_Close, 50)
        self.SPY_Close.Updated += lambda _, args: self.SPY_Close_History.Add(args)
        self.SPY_Close_EMA10.Updated += lambda _, args: self.SPY_Close_EMA10_History.Add(args)
        self.SPY_Close_EMA50.Updated += lambda _, args: self.SPY_Close_EMA50_History.Add(args)
        if self.ObjectStore.ContainsKey(self.SPY_Close_ObjectStore_Key):
            values = self.ObjectStore.Read(self.SPY_Close_ObjectStore_Key)
            self.Debug(f'{self.SPY_Close_ObjectStore_Key} key exists in object store.')
            history = pd.read_csv(StringIO(values), header=None, index_col=0, squeeze=True)
            history.index = pd.to_datetime(history.index)
            for (time, close) in history.iteritems():
                self.SPY_Close.Update(time, close)
        else:
            self.Debug(f'{self.SPY_Close_ObjectStore_Key} key does not exist in object store. Fetching history...')
            history = self.History(self.SPY, timedelta(365), Resolution.Daily).close.unstack(0).squeeze()
            for (time, close) in history.iteritems():
                self.SPY_Close.Update(time, close)
            self.ObjectStore.Save(self.SPY_Close_ObjectStore_Key, '\n'.join(reversed([f'{x.EndTime},{x.Value}' for x in self.SPY_Close_History])))

    def OnData(self, slice):
        if False:
            print('Hello World!')
        close = self.SPY_Close
        ema10 = self.SPY_Close_EMA10
        ema50 = self.SPY_Close_EMA50
        if ema10 > close and ema10 > ema50:
            self.SetHoldings(self.SPY, 1)
        elif ema10 < close and ema10 < ema50:
            self.SetHoldings(self.SPY, -1)
        elif ema10 < ema50 and self.Portfolio[self.SPY].IsLong:
            self.Liquidate(self.SPY)
        elif ema10 > ema50 and self.Portfolio[self.SPY].IsShort:
            self.Liquidate(self.SPY)