import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union
from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class SampleStrategy(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    INTERFACE_VERSION = 3
    can_short: bool = False
    minimal_roi = {'60': 0.01, '30': 0.02, '0': 0.04}
    stoploss = -0.1
    trailing_stop = False
    timeframe = '5m'
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space='sell', optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    startup_candle_count: int = 200
    order_types = {'entry': 'limit', 'exit': 'limit', 'stoploss': 'market', 'stoploss_on_exchange': False}
    order_time_in_force = {'entry': 'GTC', 'exit': 'GTC'}
    plot_config = {'main_plot': {'tema': {}, 'sar': {'color': 'white'}}, 'subplots': {'MACD': {'macd': {'color': 'blue'}, 'macdsignal': {'color': 'orange'}}, 'RSI': {'rsi': {'color': 'red'}}}}

    def informative_pairs(self):
        if False:
            while True:
                i = 10
        '\n        Define additional, informative pair/interval combinations to be cached from the exchange.\n        These pair/interval combinations are non-tradeable, unless they are part\n        of the whitelist as well.\n        For more information, please consult the documentation\n        :return: List of tuples in the format (pair, interval)\n            Sample: return [("ETH/USDT", "5m"),\n                            ("BTC/USDT", "15m"),\n                            ]\n        '
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            return 10
        '\n        Adds several different TA indicators to the given DataFrame\n\n        Performance Note: For the best performance be frugal on the number of indicators\n        you are using. Let uncomment only the indicator you are using in your strategies\n        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.\n        :param dataframe: Dataframe with data from the exchange\n        :param metadata: Additional information, like the currently traded pair\n        :return: a Dataframe with all mandatory indicators for the strategies\n        '
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['mfi'] = ta.MFI(dataframe)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']
        "\n        # first check if dataprovider is available\n        if self.dp:\n            if self.dp.runmode.value in ('live', 'dry_run'):\n                ob = self.dp.orderbook(metadata['pair'], 1)\n                dataframe['best_bid'] = ob['bids'][0][0]\n                dataframe['best_ask'] = ob['asks'][0][0]\n        "
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            while True:
                i = 10
        '\n        Based on TA indicators, populates the entry signal for the given dataframe\n        :param dataframe: DataFrame\n        :param metadata: Additional information, like the currently traded pair\n        :return: DataFrame with entry columns populated\n        '
        dataframe.loc[qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value) & (dataframe['tema'] <= dataframe['bb_middleband']) & (dataframe['tema'] > dataframe['tema'].shift(1)) & (dataframe['volume'] > 0), 'enter_long'] = 1
        dataframe.loc[qtpylib.crossed_above(dataframe['rsi'], self.short_rsi.value) & (dataframe['tema'] > dataframe['bb_middleband']) & (dataframe['tema'] < dataframe['tema'].shift(1)) & (dataframe['volume'] > 0), 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            print('Hello World!')
        '\n        Based on TA indicators, populates the exit signal for the given dataframe\n        :param dataframe: DataFrame\n        :param metadata: Additional information, like the currently traded pair\n        :return: DataFrame with exit columns populated\n        '
        dataframe.loc[qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value) & (dataframe['tema'] > dataframe['bb_middleband']) & (dataframe['tema'] < dataframe['tema'].shift(1)) & (dataframe['volume'] > 0), 'exit_long'] = 1
        dataframe.loc[qtpylib.crossed_above(dataframe['rsi'], self.exit_short_rsi.value) & (dataframe['tema'] <= dataframe['bb_middleband']) & (dataframe['tema'] > dataframe['tema'].shift(1)) & (dataframe['volume'] > 0), 'exit_short'] = 1
        return dataframe