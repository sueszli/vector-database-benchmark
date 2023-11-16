import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import CategoricalParameter

class strategy_test_v3_recursive_issue(IStrategy):
    INTERFACE_VERSION = 3
    minimal_roi = {'0': 0.04}
    stoploss = -0.1
    timeframe = '5m'
    scenario = CategoricalParameter(['no_bias', 'bias1', 'bias2'], default='bias1', space='buy')
    startup_candle_count: int = 100

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        if self.scenario.value == 'no_bias':
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        else:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=50)
        if self.scenario.value == 'bias2':
            dataframe['rsi_lookahead'] = ta.RSI(dataframe, timeperiod=50).shift(-1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            while True:
                i = 10
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if False:
            while True:
                i = 10
        return dataframe