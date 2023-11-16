from typing import Union
import numpy as np
import talib
from jesse.indicators.ma import ma
from jesse.helpers import get_candle_source, same_length
from jesse.helpers import slice_candles
from jesse.indicators.mean_ad import mean_ad
from jesse.indicators.median_ad import median_ad

def rvi(candles: np.ndarray, period: int=10, ma_len: int=14, matype: int=1, devtype: int=0, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        print('Hello World!')
    '\n    RVI - Relative Volatility Index\n    :param candles: np.ndarray\n    :param period: int - default: 10\n    :param ma_len: int - default: 14\n    :param matype: int - default: 1\n    :param devtype: int - default: 0\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    if devtype == 0:
        dev = talib.STDDEV(source, period)
    elif devtype == 1:
        dev = mean_ad(source, period, sequential=True)
    elif devtype == 2:
        dev = median_ad(source, period, sequential=True)
    diff = np.diff(source)
    diff = same_length(source, diff)
    up = np.nan_to_num(np.where(diff <= 0, 0, dev))
    down = np.nan_to_num(np.where(diff > 0, 0, dev))
    up_avg = ma(up, period=ma_len, matype=matype, sequential=True)
    down_avg = ma(down, period=ma_len, matype=matype, sequential=True)
    result = 100 * (up_avg / (up_avg + down_avg))
    return result if sequential else result[-1]