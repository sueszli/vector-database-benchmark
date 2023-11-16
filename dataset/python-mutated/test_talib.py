import pandas as pd
import talib.abstract as ta

def test_talib_bollingerbands_near_zero_values():
    if False:
        print('Hello World!')
    inputs = pd.DataFrame([{'close': 1e-07}, {'close': 1.1e-07}, {'close': 1.2e-07}, {'close': 1.3e-07}, {'close': 1.4e-07}])
    bollinger = ta.BBANDS(inputs, matype=0, timeperiod=2)
    assert bollinger['upperband'][3] != bollinger['middleband'][3]