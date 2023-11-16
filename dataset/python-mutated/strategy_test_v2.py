import talib.abstract as ta
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy

class StrategyTestV2(IStrategy):
    """
    Strategy used by tests freqtrade bot.
    Please do not modify this strategy, it's  intended for internal use only.
    Please look at the SampleStrategy in the user_data/strategy directory
    or strategy repository https://github.com/freqtrade/freqtrade-strategies
    for samples and inspiration.
    """
    INTERFACE_VERSION = 2
    minimal_roi = {'40': 0.0, '30': 0.01, '20': 0.02, '0': 0.04}
    stoploss = -0.1
    timeframe = '5m'
    order_types = {'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': False}
    startup_candle_count: int = 20
    order_time_in_force = {'entry': 'gtc', 'exit': 'gtc'}
    use_sell_signal = False
    position_adjustment_enable = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            while True:
                i = 10
        '\n        Adds several different TA indicators to the given DataFrame\n\n        Performance Note: For the best performance be frugal on the number of indicators\n        you are using. Let uncomment only the indicator you are using in your strategies\n        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.\n        :param dataframe: Dataframe with data from the exchange\n        :param metadata: Additional information, like the currently traded pair\n        :return: a Dataframe with all mandatory indicators for the strategies\n        '
        dataframe['adx'] = ta.ADX(dataframe)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            print('Hello World!')
        '\n        Based on TA indicators, populates the buy signal for the given dataframe\n        :param dataframe: DataFrame\n        :param metadata: Additional information, like the currently traded pair\n        :return: DataFrame with buy column\n        '
        dataframe.loc[(dataframe['rsi'] < 35) & (dataframe['fastd'] < 35) & (dataframe['adx'] > 30) & (dataframe['plus_di'] > 0.5) | (dataframe['adx'] > 65) & (dataframe['plus_di'] > 0.5), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            print('Hello World!')
        '\n        Based on TA indicators, populates the sell signal for the given dataframe\n        :param dataframe: DataFrame\n        :param metadata: Additional information, like the currently traded pair\n        :return: DataFrame with buy column\n        '
        dataframe.loc[(qtpylib.crossed_above(dataframe['rsi'], 70) | qtpylib.crossed_above(dataframe['fastd'], 70)) & (dataframe['adx'] > 10) & (dataframe['minus_di'] > 0) | (dataframe['adx'] > 70) & (dataframe['minus_di'] > 0.5), 'sell'] = 1
        return dataframe