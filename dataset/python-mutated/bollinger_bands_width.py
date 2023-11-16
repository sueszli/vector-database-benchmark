from typing import Union
import numpy as np
import talib
from jesse.indicators.ma import ma
from jesse.indicators.mean_ad import mean_ad
from jesse.indicators.median_ad import median_ad
from jesse.helpers import get_candle_source, slice_candles

def bollinger_bands_width(candles: np.ndarray, period: int=20, devup: float=2, devdn: float=2, matype: int=0, devtype: int=0, source_type: str='close', sequential: bool=False) -> Union[float, np.ndarray]:
    if False:
        print('Hello World!')
    '\n    BBW - Bollinger Bands Width - Bollinger Bands Bandwidth\n\n    :param candles: np.ndarray\n    :param period: int - default: 20\n    :param devup: float - default: 2\n    :param devdn: float - default: 2\n    :param matype: int - default: 0\n    :param devtype: int - default: 0\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: BollingerBands(upperband, middleband, lowerband)\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    if devtype == 0:
        dev = talib.STDDEV(source, period)
    elif devtype == 1:
        dev = mean_ad(source, period, sequential=True)
    elif devtype == 2:
        dev = median_ad(source, period, sequential=True)
    middlebands = ma(source, period=period, matype=matype, sequential=True)
    upperbands = middlebands + devup * dev
    lowerbands = middlebands - devdn * dev
    if sequential:
        return (upperbands - lowerbands) / middlebands
    else:
        return (upperbands[-1] - lowerbands[-1]) / middlebands[-1]