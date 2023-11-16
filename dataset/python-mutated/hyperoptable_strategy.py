from pandas import DataFrame
from strategy_test_v3 import StrategyTestV3
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import BooleanParameter, DecimalParameter, IntParameter, RealParameter

class HyperoptableStrategy(StrategyTestV3):
    """
    Default Strategy provided by freqtrade bot.
    Please do not modify this strategy, it's  intended for internal use only.
    Please look at the SampleStrategy in the user_data/strategy directory
    or strategy repository https://github.com/freqtrade/freqtrade-strategies
    for samples and inspiration.
    """
    buy_params = {'buy_rsi': 35}
    sell_params = {'sell_rsi': 74, 'sell_minusdi': 0.4}
    buy_plusdi = RealParameter(low=0, high=1, default=0.5, space='buy')
    sell_rsi = IntParameter(low=50, high=100, default=70, space='sell')
    sell_minusdi = DecimalParameter(low=0, high=1, default=0.5001, decimals=3, space='sell', load=False)
    protection_enabled = BooleanParameter(default=True)
    protection_cooldown_lookback = IntParameter([0, 50], default=30)
    plot_config = {'main_plot': {}}

    @property
    def protections(self):
        if False:
            print('Hello World!')
        prot = []
        if self.protection_enabled.value:
            prot.append({'method': 'CooldownPeriod', 'stop_duration_candles': self.protection_cooldown_lookback.value})
        return prot
    bot_loop_started = False
    bot_started = False

    def bot_loop_start(self):
        if False:
            i = 10
            return i + 15
        self.bot_loop_started = True

    def bot_start(self, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Parameters can also be defined here ...\n        '
        self.bot_started = True
        self.buy_rsi = IntParameter([0, 50], default=30, space='buy')

    def informative_pairs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Define additional, informative pair/interval combinations to be cached from the exchange.\n        These pair/interval combinations are non-tradeable, unless they are part\n        of the whitelist as well.\n        For more information, please consult the documentation\n        :return: List of tuples in the format (pair, interval)\n            Sample: return [("ETH/USDT", "5m"),\n                            ("BTC/USDT", "15m"),\n                            ]\n        '
        return []

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            i = 10
            return i + 15
        '\n        Based on TA indicators, populates the buy signal for the given dataframe\n        :param dataframe: DataFrame\n        :param metadata: Additional information, like the currently traded pair\n        :return: DataFrame with buy column\n        '
        dataframe.loc[(dataframe['rsi'] < self.buy_rsi.value) & (dataframe['fastd'] < 35) & (dataframe['adx'] > 30) & (dataframe['plus_di'] > self.buy_plusdi.value) | (dataframe['adx'] > 65) & (dataframe['plus_di'] > self.buy_plusdi.value), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            i = 10
            return i + 15
        '\n        Based on TA indicators, populates the sell signal for the given dataframe\n        :param dataframe: DataFrame\n        :param metadata: Additional information, like the currently traded pair\n        :return: DataFrame with sell column\n        '
        dataframe.loc[(qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value) | qtpylib.crossed_above(dataframe['fastd'], 70)) & (dataframe['adx'] > 10) & (dataframe['minus_di'] > 0) | (dataframe['adx'] > 70) & (dataframe['minus_di'] > self.sell_minusdi.value), 'sell'] = 1
        return dataframe