from collections import namedtuple
import numpy as np
import talib
from jesse.helpers import get_candle_source
from jesse.helpers import slice_candles
IQ = namedtuple('IQ', ['inphase', 'quadrature'])

def ht_phasor(candles: np.ndarray, source_type: str='close', sequential: bool=False) -> IQ:
    if False:
        for i in range(10):
            print('nop')
    '\n    HT_PHASOR - Hilbert Transform - Phasor Components\n\n    :param candles: np.ndarray\n    :param source_type: str - default: "close"\n    :param sequential: bool - default: False\n\n    :return: IQ(inphase, quadrature)\n    '
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    (inphase, quadrature) = talib.HT_PHASOR(source)
    if sequential:
        return IQ(inphase, quadrature)
    else:
        return IQ(inphase[-1], quadrature[-1])