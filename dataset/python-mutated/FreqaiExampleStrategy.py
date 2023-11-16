import logging
from functools import reduce
from typing import Dict
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from freqtrade.strategy import CategoricalParameter, IStrategy
logger = logging.getLogger(__name__)

class FreqaiExampleStrategy(IStrategy):
    """
    Example strategy showing how the user connects their own
    IFreqaiModel to the strategy.

    Warning! This is a showcase of functionality,
    which means that it is designed to show various functions of FreqAI
    and it runs on all computers. We use this showcase to help users
    understand how to build a strategy, and we use it as a benchmark
    to help debug possible problems.

    This means this is *not* meant to be run live in production.
    """
    minimal_roi = {'0': 0.1, '240': -1}
    plot_config = {'main_plot': {}, 'subplots': {'&-s_close': {'&-s_close': {'color': 'blue'}}, 'do_predict': {'do_predict': {'color': 'brown'}}}}
    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 40
    can_short = True
    std_dev_multiplier_buy = CategoricalParameter([0.75, 1, 1.25, 1.5, 1.75], default=1.25, space='buy', optimize=True)
    std_dev_multiplier_sell = CategoricalParameter([0.75, 1, 1.25, 1.5, 1.75], space='sell', default=1.25, optimize=True)

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs) -> DataFrame:
        if False:
            while True:
                i = 10
        '\n        *Only functional with FreqAI enabled strategies*\n        This function will automatically expand the defined features on the config defined\n        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and\n        `include_corr_pairs`. In other words, a single feature defined in this function\n        will automatically expand to a total of\n        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *\n        `include_corr_pairs` numbers of features added to the model.\n\n        All features must be prepended with `%` to be recognized by FreqAI internals.\n\n        Access metadata such as the current pair/timeframe with:\n\n        `metadata["pair"]` `metadata["tf"]`\n\n        More details on how these config defined parameters accelerate feature engineering\n        in the documentation at:\n\n        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters\n\n        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features\n\n        :param dataframe: strategy dataframe which will receive the features\n        :param period: period of the indicator - usage example:\n        :param metadata: metadata of current pair\n        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)\n        '
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
        '\n        *Only functional with FreqAI enabled strategies*\n        This function will automatically expand the defined features on the config defined\n        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.\n        In other words, a single feature defined in this function\n        will automatically expand to a total of\n        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`\n        numbers of features added to the model.\n\n        Features defined here will *not* be automatically duplicated on user defined\n        `indicator_periods_candles`\n\n        All features must be prepended with `%` to be recognized by FreqAI internals.\n\n        Access metadata such as the current pair/timeframe with:\n\n        `metadata["pair"]` `metadata["tf"]`\n\n        More details on how these config defined parameters accelerate feature engineering\n        in the documentation at:\n\n        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters\n\n        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features\n\n        :param dataframe: strategy dataframe which will receive the features\n        :param metadata: metadata of current pair\n        dataframe["%-pct-change"] = dataframe["close"].pct_change()\n        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)\n        '
        dataframe['%-pct-change'] = dataframe['close'].pct_change()
        dataframe['%-raw_volume'] = dataframe['volume']
        dataframe['%-raw_price'] = dataframe['close']
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        if False:
            i = 10
            return i + 15
        '\n        *Only functional with FreqAI enabled strategies*\n        This optional function will be called once with the dataframe of the base timeframe.\n        This is the final function to be called, which means that the dataframe entering this\n        function will contain all the features and columns created by all other\n        freqai_feature_engineering_* functions.\n\n        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).\n        This function is a good place for any feature that should not be auto-expanded upon\n        (e.g. day of the week).\n\n        All features must be prepended with `%` to be recognized by FreqAI internals.\n\n        Access metadata such as the current pair with:\n\n        `metadata["pair"]`\n\n        More details about feature engineering available:\n\n        https://www.freqtrade.io/en/latest/freqai-feature-engineering\n\n        :param dataframe: strategy dataframe which will receive the features\n        :param metadata: metadata of current pair\n        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7\n        '
        dataframe['%-day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['%-hour_of_day'] = dataframe['date'].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        if False:
            while True:
                i = 10
        '\n        *Only functional with FreqAI enabled strategies*\n        Required function to set the targets for the model.\n        All targets must be prepended with `&` to be recognized by the FreqAI internals.\n\n        Access metadata such as the current pair with:\n\n        `metadata["pair"]`\n\n        More details about feature engineering available:\n\n        https://www.freqtrade.io/en/latest/freqai-feature-engineering\n\n        :param dataframe: strategy dataframe which will receive the targets\n        :param metadata: metadata of current pair\n        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]\n        '
        dataframe['&-s_close'] = dataframe['close'].shift(-self.freqai_info['feature_parameters']['label_period_candles']).rolling(self.freqai_info['feature_parameters']['label_period_candles']).mean() / dataframe['close'] - 1
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            i = 10
            return i + 15
        dataframe = self.freqai.start(dataframe, metadata, self)
        for val in self.std_dev_multiplier_buy.range:
            dataframe[f'target_roi_{val}'] = dataframe['&-s_close_mean'] + dataframe['&-s_close_std'] * val
        for val in self.std_dev_multiplier_sell.range:
            dataframe[f'sell_roi_{val}'] = dataframe['&-s_close_mean'] - dataframe['&-s_close_std'] * val
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        enter_long_conditions = [df['do_predict'] == 1, df['&-s_close'] > df[f'target_roi_{self.std_dev_multiplier_buy.value}']]
        if enter_long_conditions:
            df.loc[reduce(lambda x, y: x & y, enter_long_conditions), ['enter_long', 'enter_tag']] = (1, 'long')
        enter_short_conditions = [df['do_predict'] == 1, df['&-s_close'] < df[f'sell_roi_{self.std_dev_multiplier_sell.value}']]
        if enter_short_conditions:
            df.loc[reduce(lambda x, y: x & y, enter_short_conditions), ['enter_short', 'enter_tag']] = (1, 'short')
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        exit_long_conditions = [df['do_predict'] == 1, df['&-s_close'] < df[f'sell_roi_{self.std_dev_multiplier_sell.value}'] * 0.25]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), 'exit_long'] = 1
        exit_short_conditions = [df['do_predict'] == 1, df['&-s_close'] > df[f'target_roi_{self.std_dev_multiplier_buy.value}'] * 0.25]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), 'exit_short'] = 1
        return df

    def get_ticker_indicator(self):
        if False:
            for i in range(10):
                print('nop')
        return int(self.config['timeframe'][:-1])

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, current_time, entry_tag, side: str, **kwargs) -> bool:
        if False:
            return 10
        (df, _) = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()
        if side == 'long':
            if rate > last_candle['close'] * (1 + 0.0025):
                return False
        elif rate < last_candle['close'] * (1 - 0.0025):
            return False
        return True