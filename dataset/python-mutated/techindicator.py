__author__ = 'saeedamen & mhockenberger'
import pandas as pd
import numpy as np
from collections import OrderedDict
from findatapy.timeseries import Calculations
from findatapy.util.loggermanager import LoggerManager

class TechIndicator(object):
    """Calculates various technical indicators and associated trading signals.

    Signals can also be filtered to allow only longs or short and also to flip signals.

    It is possible to create subclasses of
    TechIndicator to implement your own custom signals and technical indicators.

    At present it implements the following technical indicators (and associated signals)
        SMA - simple moving averages - buy if spot above SMA, sell if below)
        EMA - exponential moving average - buy if spot spot above EMA, sell if below)
        ROC - rate of change - buy if rate of change positive, sell if negative
        polarity - buy if input is positive, sell if negative
        SMA2 - double simple moving average - buy if faster SMA above slower SMA, sell if below
        RSI - relative strength indicator - buy if RSI exits overbought, sell if RSI exits oversold
        BB - Bollinger bands - buy if spot above upper Bollinger band, sell if below lower Bollinger Band
        ATR - Average True Range - useful for creating stop/take profit levels
        long_only - long only signal

    """

    def __init__(self):
        if False:
            return 10
        self._techind = None
        self._signal = None

    def create_tech_ind(self, data_frame_non_nan, name, tech_params, data_frame_non_nan_early=None):
        if False:
            while True:
                i = 10
        logger = LoggerManager().getLogger(__name__)
        self._signal = None
        self._techind = None
        if tech_params.fillna:
            data_frame = data_frame_non_nan.fillna(method='ffill')
        else:
            data_frame = data_frame_non_nan
        if data_frame_non_nan_early is not None:
            data_frame_early = data_frame_non_nan_early.fillna(method='ffill')
        if name == 'SMA':
            if data_frame_non_nan_early is not None:
                if pd.__version__ < '0.17':
                    rolling_sum = pd.rolling_sum(data_frame.shift(1).rolling, window=tech_params.sma_period - 1)
                else:
                    rolling_sum = data_frame.shift(1).rolling(center=False, window=tech_params.sma_period - 1).sum()
                rolling_sum = rolling_sum + data_frame_early
                self._techind = rolling_sum / tech_params.sma_period
                narray = np.where(data_frame_early > self._techind, 1, -1)
            else:
                if pd.__version__ < '0.17':
                    self._techind = pd.rolling_sum(data_frame, window=tech_params.sma_period)
                else:
                    self._techind = data_frame.rolling(window=tech_params.sma_period, center=False).mean()
                narray = np.where(data_frame > self._techind, 1, -1)
            self._signal = pd.DataFrame(index=data_frame.index, data=narray)
            self._signal.iloc[0:tech_params.sma_period] = np.nan
            self._signal.columns = [x + ' SMA Signal' for x in data_frame.columns.values]
            self._techind.columns = [x + ' SMA' for x in data_frame.columns.values]
        elif name == 'EMA':
            self._techind = data_frame.ewm(ignore_na=False, span=tech_params.ema_period, min_periods=0, adjust=True).mean()
            narray = np.where(data_frame > self._techind, 1, -1)
            self._signal = pd.DataFrame(index=data_frame.index, data=narray)
            self._signal.iloc[0:tech_params.ema_period] = np.nan
            self._signal.columns = [x + ' EMA Signal' for x in data_frame.columns.values]
            self._techind.columns = [x + ' EMA' for x in data_frame.columns.values]
        elif name == 'ROC':
            if data_frame_non_nan_early is not None:
                self._techind = data_frame_early / data_frame.shift(tech_params.roc_period) - 1
            else:
                self._techind = data_frame / data_frame.shift(tech_params.roc_period) - 1
            narray = np.where(self._techind > 0, 1, -1)
            self._signal = pd.DataFrame(index=data_frame.index, data=narray)
            self._signal.iloc[0:tech_params.roc_period] = np.nan
            self._signal.columns = [x + ' ROC Signal' for x in data_frame.columns.values]
            self._techind.columns = [x + ' ROC' for x in data_frame.columns.values]
        elif name == 'polarity':
            self._techind = data_frame
            narray = np.where(self._techind > 0, 1, -1)
            self._signal = pd.DataFrame(index=data_frame.index, data=narray)
            self._signal.columns = [x + ' Polarity Signal' for x in data_frame.columns.values]
            self._techind.columns = [x + ' Polarity' for x in data_frame.columns.values]
        elif name == 'SMA2':
            sma = data_frame.rolling(window=tech_params.sma_period, center=False).mean()
            sma2 = data_frame.rolling(window=tech_params.sma2_period, center=False).mean()
            narray = np.where(sma > sma2, 1, -1)
            self._signal = pd.DataFrame(index=data_frame.index, data=narray)
            self._signal.columns = [x + ' SMA2 Signal' for x in data_frame.columns.values]
            sma.columns = [x + ' SMA' for x in data_frame.columns.values]
            sma2.columns = [x + ' SMA2' for x in data_frame.columns.values]
            most = max(tech_params.sma_period, tech_params.sma2_period)
            self._signal.iloc[0:most] = np.nan
            self._techind = pd.concat([sma, sma2], axis=1)
        elif name in ['RSI']:
            delta = data_frame.diff()
            delta = delta[1:]
            (up, down) = (delta.copy(), delta.copy())
            up[up < 0] = 0
            down[down > 0] = 0
            roll_up1 = pd.stats.moments.ewma(up, tech_params.rsi_period)
            roll_down1 = pd.stats.moments.ewma(down.abs(), tech_params.rsi_period)
            RS1 = roll_up1 / roll_down1
            RSI1 = 100.0 - 100.0 / (1.0 + RS1)
            roll_up2 = up.rolling(window=tech_params.rsi_period, center=False).mean()
            roll_down2 = down.abs().rolling(window=tech_params.rsi_period, center=False).mean()
            RS2 = roll_up2 / roll_down2
            RSI2 = 100.0 - 100.0 / (1.0 + RS2)
            self._techind = RSI2
            self._techind.columns = [x + ' RSI' for x in data_frame.columns.values]
            signal = data_frame.copy()
            sells = (signal.shift(-1) < tech_params.rsi_lower) & (signal > tech_params.rsi_lower)
            buys = (signal.shift(-1) > tech_params.rsi_upper) & (signal < tech_params.rsi_upper)
            signal[buys] = 1
            signal[sells] = -1
            signal[~(buys | sells)] = np.nan
            signal = signal.fillna(method='ffill')
            self._signal = signal
            self._signal.iloc[0:tech_params.rsi_period] = np.nan
            self._signal.columns = [x + ' RSI Signal' for x in data_frame.columns.values]
        elif name in ['BB']:
            mid = data_frame.rolling(center=False, window=tech_params.bb_period).mean()
            mid.columns = [x + ' BB Mid' for x in data_frame.columns.values]
            std_dev = data_frame.rolling(center=False, window=tech_params.bb_period).std()
            BB_std = tech_params.bb_mult * std_dev
            lower = pd.DataFrame(data=mid.values - BB_std.values, index=mid.index, columns=data_frame.columns)
            upper = pd.DataFrame(data=mid.values + BB_std.values, index=mid.index, columns=data_frame.columns)
            signal = data_frame.copy()
            buys = signal > upper
            sells = signal < lower
            signal[buys] = 1
            signal[sells] = -1
            signal[~(buys | sells)] = np.nan
            signal = signal.fillna(method='ffill')
            self._signal = signal
            self._signal.iloc[0:tech_params.bb_period] = np.nan
            self._signal.columns = [x + ' ' + name + ' Signal' for x in data_frame.columns.values]
            lower.columns = [x + ' BB Lower' for x in data_frame.columns.values]
            upper.columns = [x + ' BB Mid' for x in data_frame.columns.values]
            upper.columns = [x + ' BB Lower' for x in data_frame.columns.values]
            self._techind = pd.concat([lower, mid, upper], axis=1)
        elif name == 'long-only':
            self._techind = data_frame
            narray = np.ones((len(data_frame.index), len(data_frame.columns)))
            self._signal = pd.DataFrame(index=data_frame.index, data=narray)
            self._signal.columns = [x + ' Long Only Signal' for x in data_frame.columns.values]
            self._techind.columns = [x + ' Long Only' for x in data_frame.columns.values]
        elif name == 'short-only':
            self._techind = data_frame
            narray = np.ones((len(data_frame.index), len(data_frame.columns)))
            self._signal = pd.DataFrame(index=data_frame.index, data=narray)
            self._signal.columns = [x + ' Short Only Signal' for x in data_frame.columns.values]
            self._techind.columns = [x + ' Short Only' for x in data_frame.columns.values]
        elif name == 'ATR':
            asset_name = list(OrderedDict.fromkeys([x.split('.')[0] for x in data_frame.columns]))
            df = []
            for a in asset_name:
                close = [a + '.close']
                low = [a + '.low']
                high = [a + '.high']
                if not tech_params.fillna:
                    data_frame_short = data_frame[[close[0], low[0], high[0]]]
                    data_frame_short = data_frame_short.dropna()
                else:
                    data_frame_short = data_frame
                prev_close = data_frame_short[close].shift(1)
                c1 = data_frame_short[high].values - data_frame_short[low].values
                c2 = np.abs(data_frame_short[high].values - prev_close.values)
                c3 = np.abs(data_frame_short[low].values - prev_close.values)
                true_range = np.max((c1, c2, c3), axis=0)
                true_range = pd.DataFrame(index=data_frame_short.index, data=true_range, columns=[close[0] + ' True Range'])
                if not tech_params.fillna:
                    true_range = true_range.reindex(data_frame.index, fill_value=np.nan)
                df.append(true_range)
            calc = Calculations()
            true_range = calc.join(df, how='outer')
            self._techind = true_range.rolling(window=tech_params.atr_period, center=False).mean()
            self._techind.columns = [x + '.close ATR' for x in asset_name]
        elif name in ['VWAP']:
            asset_name = list(OrderedDict.fromkeys([x.split('.')[0] for x in data_frame.columns]))
            df = []
            for a in asset_name:
                high = [a + '.high']
                low = [a + '.low']
                close = [a + '.close']
                volume = [a + '.volume']
                if not tech_params.fillna:
                    df_mod = data_frame[[high[0], low[0], close[0], volume[0]]]
                    df_mod.dropna(inplace=True)
                else:
                    df_mod = data_frame
                l = df_mod[low].values
                h = df_mod[high].values
                c = df_mod[close].values
                v = df_mod[volume].values
                vwap = np.cumsum((h + l + c) / 3 * v) / np.cumsum(v)
                vwap = pd.DataFrame(index=df_mod.index, data=vwap, columns=[close[0] + ' VWAP'])
                print(vwap.columns)
                if not tech_params.fillna:
                    vwap = vwap.reindex(data_frame.index, fill_value=np.nan)
                df.append(vwap)
            calc = Calculations()
            vwap = calc.join(df, how='outer')
            self._techind = vwap
            self._techind.columns = [x + '.close VWAP' for x in asset_name]
        self.create_custom_tech_ind(data_frame_non_nan, name, tech_params, data_frame_non_nan_early)
        if hasattr(tech_params, 'only_allow_longs'):
            self._signal[self._signal < 0] = 0
        if hasattr(tech_params, 'only_allow_shorts'):
            self._signal[self._signal > 0] = 0
        if hasattr(tech_params, 'signal_mult'):
            self._signal = self._signal * tech_params.signal_mult
        if hasattr(tech_params, 'strip_signal_name'):
            if tech_params.strip_signal_name:
                self._signal.columns = data_frame.columns
        return (self._techind, self._signal)

    def create_custom_tech_ind(self, data_frame_non_nan, name, tech_params, data_frame_non_nan_early):
        if False:
            return 10
        return

    def get_techind(self):
        if False:
            i = 10
            return i + 15
        return self._techind

    def get_signal(self):
        if False:
            for i in range(10):
                print('nop')
        return self._signal

class TechParams(object):
    """Holds parameters for calculation of technical indicators.

    """

    def __init__(self, fillna=True, atr_period=14, sma_period=20, green_n=4, green_count=9, red_n=2, red_count=13):
        if False:
            i = 10
            return i + 15
        self.fillna = fillna
        self.atr_period = atr_period
        self.sma_period = sma_period
        self.green_n = green_n
        self.green_count = green_count
        self.red_n = red_n
        self.red_count = red_count

    @property
    def fillna(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__fillna

    @fillna.setter
    def fillna(self, fillna):
        if False:
            while True:
                i = 10
        self.__fillna = fillna

    @property
    def atr_period(self):
        if False:
            i = 10
            return i + 15
        return self.__atr_period

    @atr_period.setter
    def atr_period(self, atr_period):
        if False:
            return 10
        self.__atr_period = atr_period

    @property
    def sma_period(self):
        if False:
            print('Hello World!')
        return self.__sma_period

    @sma_period.setter
    def sma_period(self, sma_period):
        if False:
            print('Hello World!')
        self.__sma_period = sma_period

    @property
    def green_n(self):
        if False:
            return 10
        return self.__green_n

    @green_n.setter
    def green_n(self, green_n):
        if False:
            for i in range(10):
                print('nop')
        self.__green_n = green_n

    @property
    def green_count(self):
        if False:
            while True:
                i = 10
        return self.__green_count

    @green_count.setter
    def green_count(self, green_count):
        if False:
            while True:
                i = 10
        self.__green_count = green_count

    @property
    def red_n(self):
        if False:
            i = 10
            return i + 15
        return self.__red_n

    @red_n.setter
    def red_n(self, red_n):
        if False:
            while True:
                i = 10
        self.__red_n = red_n

    @property
    def red_count(self):
        if False:
            print('Hello World!')
        return self.__red_count

    @property
    def only_allow_shorts(self):
        if False:
            print('Hello World!')
        return self.__only_allow_shorts

    @property
    def only_allow_longs(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__only_allow_longs

    @red_count.setter
    def red_count(self, red_count):
        if False:
            return 10
        self.__red_count = red_count

    @only_allow_longs.setter
    def only_allow_longs(self, only_allow_longs):
        if False:
            return 10
        if hasattr(self, 'only_allow_shorts'):
            raise Exception('Attribute only_allow_shorts is already defined and it is not compatible with attribute only_allow_longs')
        self.__only_allow_longs = only_allow_longs

    @only_allow_shorts.setter
    def only_allow_shorts(self, only_allow_shorts):
        if False:
            return 10
        if hasattr(self, 'only_allow_longs'):
            raise Exception('Attribute only_allow_longs is already defined and it is not compatible with attribute only_allow_shorts')
        self.__only_allow_shorts = only_allow_shorts