from AlgorithmImports import *
import talib

class CalibratedResistanceAtmosphericScrubbers(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2020, 1, 2)
        self.SetEndDate(2020, 1, 6)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Hour)
        self.rolling_window = pd.DataFrame()
        self.dema_period = 3
        self.sma_period = 3
        self.wma_period = 3
        self.window_size = self.dema_period * 2
        self.SetWarmUp(self.window_size)

    def OnData(self, data):
        if False:
            return 10
        if 'SPY' not in data.Bars:
            return
        close = data['SPY'].Close
        if self.IsWarmingUp:
            row = pd.DataFrame({'close': [close]}, index=[data.Time])
            self.rolling_window = self.rolling_window.append(row).iloc[-self.window_size:]
            if self.rolling_window.shape[0] == self.window_size:
                closes = self.rolling_window['close'].values
                self.rolling_window['DEMA'] = talib.DEMA(closes, self.dema_period)
                self.rolling_window['EMA'] = talib.EMA(closes, self.sma_period)
                self.rolling_window['WMA'] = talib.WMA(closes, self.wma_period)
            return
        closes = np.append(self.rolling_window['close'].values, close)[-self.window_size:]
        row = pd.DataFrame({'close': close, 'DEMA': talib.DEMA(closes, self.dema_period)[-1], 'EMA': talib.EMA(closes, self.sma_period)[-1], 'WMA': talib.WMA(closes, self.wma_period)[-1]}, index=[data.Time])
        self.rolling_window = self.rolling_window.append(row).iloc[-self.window_size:]

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        self.Log(f'\nRolling Window:\n{self.rolling_window.to_string()}\n')
        self.Log(f'\nLatest Values:\n{self.rolling_window.iloc[-1].to_string()}\n')