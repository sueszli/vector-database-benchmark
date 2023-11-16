from pandas import DataFrame
from freqtrade.strategy import IStrategy

class TestStrategyLegacyV1(IStrategy):
    minimal_roi = {'40': 0.0, '30': 0.01, '20': 0.02, '0': 0.04}
    stoploss = -0.1
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        if False:
            while True:
                i = 10
        return dataframe