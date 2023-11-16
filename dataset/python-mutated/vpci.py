from collections import namedtuple
import numpy as np
import talib
from jesse.helpers import slice_candles
VPCI = namedtuple('VPCI', ['vpci', 'vpcis'])

def vpci(candles: np.ndarray, short_range: int=5, long_range: int=25, sequential: bool=False) -> VPCI:
    if False:
        print('Hello World!')
    '\n    VPCI - Volume Price Confirmation Indicator\n\n    :param candles: np.ndarray\n    :param short_range: int - default: 5\n    :param long_range: int - default: 25\n    :param sequential: bool - default: False\n\n    :return: float | np.ndarray\n    '
    candles = slice_candles(candles, sequential)
    vwma_long = talib.SMA(candles[:, 2] * candles[:, 5], long_range) / talib.SMA(candles[:, 5], long_range)
    VPC = vwma_long - talib.SMA(candles[:, 2], long_range)
    vwma_short = talib.SMA(candles[:, 2] * candles[:, 5], short_range) / talib.SMA(candles[:, 5], short_range)
    VPR = vwma_short / talib.SMA(candles[:, 2], short_range)
    VM = talib.SMA(candles[:, 5], short_range) / talib.SMA(candles[:, 5], long_range)
    VPCI_val = VPC * VPR * VM
    VPCIS = talib.SMA(VPCI_val * candles[:, 5], short_range) / talib.SMA(candles[:, 5], short_range)
    if sequential:
        return VPCI(VPCI_val, VPCIS)
    else:
        return VPCI(VPCI_val[-1], VPCIS[-1])