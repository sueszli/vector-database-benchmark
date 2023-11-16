import logging
from typing import Dict
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from freqtrade.strategy import IntParameter, IStrategy, merge_informative_pair
logger = logging.getLogger(__name__)

class FreqaiExampleHybridStrategy(IStrategy):
    """
    Example of a hybrid FreqAI strat, designed to illustrate how a user may employ
    FreqAI to bolster a typical Freqtrade strategy.

    Launching this strategy would be:

    freqtrade trade --strategy FreqaiExampleHybridStrategy --strategy-path freqtrade/templates
    --freqaimodel CatboostClassifier --config config_examples/config_freqai.example.json

    or the user simply adds this to their config:

    "freqai": {
        "enabled": true,
        "purge_old_models": 2,
        "train_period_days": 15,
        "identifier": "uniqe-id",
        "feature_parameters": {
            "include_timeframes": [
                "3m",
                "15m",
                "1h"
            ],
            "include_corr_pairlist": [
                "BTC/USDT",
                "ETH/USDT"
            ],
            "label_period_candles": 20,
            "include_shifted_candles": 2,
            "DI_threshold": 0.9,
            "weight_factor": 0.9,
            "principal_component_analysis": false,
            "use_SVM_to_remove_outliers": true,
            "indicator_periods_candles": [10, 20]
        },
        "data_split_parameters": {
            "test_size": 0,
            "random_state": 1
        },
        "model_training_parameters": {
            "n_estimators": 800
        }
    },

    Thanks to @smarmau and @johanvulgt for developing and sharing the strategy.
    """
    minimal_roi = {'60': 0.01, '30': 0.02, '0': 0.04}
    plot_config = {'main_plot': {'tema': {}}, 'subplots': {'MACD': {'macd': {'color': 'blue'}, 'macdsignal': {'color': 'orange'}}, 'RSI': {'rsi': {'color': 'red'}}, 'Up_or_down': {'&s-up_or_down': {'color': 'green'}}}}
    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 30
    can_short = True
    buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space='sell', optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs) -> DataFrame:
        if False:
            i = 10
            return i + 15
        '\n        *Only functional with FreqAI enabled strategies*\n        This function will automatically expand the defined features on the config defined\n        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and\n        `include_corr_pairs`. In other words, a single feature defined in this function\n        will automatically expand to a total of\n        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *\n        `include_corr_pairs` numbers of features added to the model.\n\n        All features must be prepended with `%` to be recognized by FreqAI internals.\n\n        More details on how these config defined parameters accelerate feature engineering\n        in the documentation at:\n\n        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters\n\n        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features\n\n        :param dataframe: strategy dataframe which will receive the features\n        :param period: period of the indicator - usage example:\n        :param metadata: metadata of current pair\n        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)\n        '
        dataframe['%-rsi-period'] = ta.RSI(dataframe, timeperiod=period)
        dataframe['%-mfi-period'] = ta.MFI(dataframe, timeperiod=period)
        dataframe['%-adx-period'] = ta.ADX(dataframe, timeperiod=period)
        dataframe['%-sma-period'] = ta.SMA(dataframe, timeperiod=period)
        dataframe['%-ema-period'] = ta.EMA(dataframe, timeperiod=period)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2.2)
        dataframe['bb_lowerband-period'] = bollinger['lower']
        dataframe['bb_middleband-period'] = bollinger['mid']
        dataframe['bb_upperband-period'] = bollinger['upper']
        dataframe['%-bb_width-period'] = (dataframe['bb_upperband-period'] - dataframe['bb_lowerband-period']) / dataframe['bb_middleband-period']
        dataframe['%-close-bb_lower-period'] = dataframe['close'] / dataframe['bb_lowerband-period']
        dataframe['%-roc-period'] = ta.ROC(dataframe, timeperiod=period)
        dataframe['%-relative_volume-period'] = dataframe['volume'] / dataframe['volume'].rolling(period).mean()
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        '\n        *Only functional with FreqAI enabled strategies*\n        This function will automatically expand the defined features on the config defined\n        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.\n        In other words, a single feature defined in this function\n        will automatically expand to a total of\n        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`\n        numbers of features added to the model.\n\n        Features defined here will *not* be automatically duplicated on user defined\n        `indicator_periods_candles`\n\n        All features must be prepended with `%` to be recognized by FreqAI internals.\n\n        More details on how these config defined parameters accelerate feature engineering\n        in the documentation at:\n\n        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters\n\n        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features\n\n        :param dataframe: strategy dataframe which will receive the features\n        :param metadata: metadata of current pair\n        dataframe["%-pct-change"] = dataframe["close"].pct_change()\n        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)\n        '
        dataframe['%-pct-change'] = dataframe['close'].pct_change()
        dataframe['%-raw_volume'] = dataframe['volume']
        dataframe['%-raw_price'] = dataframe['close']
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        if False:
            return 10
        '\n        *Only functional with FreqAI enabled strategies*\n        This optional function will be called once with the dataframe of the base timeframe.\n        This is the final function to be called, which means that the dataframe entering this\n        function will contain all the features and columns created by all other\n        freqai_feature_engineering_* functions.\n\n        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).\n        This function is a good place for any feature that should not be auto-expanded upon\n        (e.g. day of the week).\n\n        All features must be prepended with `%` to be recognized by FreqAI internals.\n\n        More details about feature engineering available:\n\n        https://www.freqtrade.io/en/latest/freqai-feature-engineering\n\n        :param dataframe: strategy dataframe which will receive the features\n        :param metadata: metadata of current pair\n        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7\n        '
        dataframe['%-day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['%-hour_of_day'] = dataframe['date'].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        if False:
            i = 10
            return i + 15
        '\n        *Only functional with FreqAI enabled strategies*\n        Required function to set the targets for the model.\n        All targets must be prepended with `&` to be recognized by the FreqAI internals.\n\n        More details about feature engineering available:\n\n        https://www.freqtrade.io/en/latest/freqai-feature-engineering\n\n        :param dataframe: strategy dataframe which will receive the targets\n        :param metadata: metadata of current pair\n        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]\n        '
        self.freqai.class_names = ['down', 'up']
        dataframe['&s-up_or_down'] = np.where(dataframe['close'].shift(-50) > dataframe['close'], 'up', 'down')
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            return 10
        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe['rsi'] = ta.RSI(dataframe)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if False:
            i = 10
            return i + 15
        df.loc[qtpylib.crossed_above(df['rsi'], self.buy_rsi.value) & (df['tema'] <= df['bb_middleband']) & (df['tema'] > df['tema'].shift(1)) & (df['volume'] > 0) & (df['do_predict'] == 1) & (df['&s-up_or_down'] == 'up'), 'enter_long'] = 1
        df.loc[qtpylib.crossed_above(df['rsi'], self.short_rsi.value) & (df['tema'] > df['bb_middleband']) & (df['tema'] < df['tema'].shift(1)) & (df['volume'] > 0) & (df['do_predict'] == 1) & (df['&s-up_or_down'] == 'down'), 'enter_short'] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if False:
            while True:
                i = 10
        df.loc[qtpylib.crossed_above(df['rsi'], self.sell_rsi.value) & (df['tema'] > df['bb_middleband']) & (df['tema'] < df['tema'].shift(1)) & (df['volume'] > 0), 'exit_long'] = 1
        df.loc[qtpylib.crossed_above(df['rsi'], self.exit_short_rsi.value) & (df['tema'] <= df['bb_middleband']) & (df['tema'] > df['tema'].shift(1)) & (df['volume'] > 0), 'exit_short'] = 1
        return df