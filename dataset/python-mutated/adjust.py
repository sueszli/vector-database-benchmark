from bisect import bisect_right
import numpy as np
from rqalpha.utils.datetime_func import convert_date_to_int
PRICE_FIELDS = {'open', 'close', 'high', 'low', 'limit_up', 'limit_down', 'acc_net_value', 'unit_net_value'}
FIELDS_REQUIRE_ADJUSTMENT = set(list(PRICE_FIELDS) + ['volume'])

def _factor_for_date(dates, factors, d):
    if False:
        return 10
    pos = bisect_right(dates, d)
    return factors[pos - 1]

def adjust_bars(bars, ex_factors, fields, adjust_type, adjust_orig):
    if False:
        i = 10
        return i + 15
    if ex_factors is None or len(bars) == 0:
        return bars
    dates = ex_factors['start_date']
    ex_cum_factors = ex_factors['ex_cum_factor']
    if adjust_type == 'pre':
        adjust_orig_dt = np.uint64(convert_date_to_int(adjust_orig))
        base_adjust_rate = _factor_for_date(dates, ex_cum_factors, adjust_orig_dt)
    else:
        base_adjust_rate = 1.0
    start_date = bars['datetime'][0]
    end_date = bars['datetime'][-1]
    if _factor_for_date(dates, ex_cum_factors, start_date) == base_adjust_rate and _factor_for_date(dates, ex_cum_factors, end_date) == base_adjust_rate:
        return bars
    factors = ex_cum_factors.take(dates.searchsorted(bars['datetime'], side='right') - 1)
    bars = np.copy(bars)
    factors /= base_adjust_rate
    if isinstance(fields, str):
        if fields in PRICE_FIELDS:
            bars[fields] *= factors
            return bars
        elif fields == 'volume':
            bars[fields] *= 1 / factors
            return bars
        return bars
    for f in bars.dtype.names:
        if f in PRICE_FIELDS:
            bars[f] *= factors
        elif f == 'volume':
            bars[f] *= 1 / factors
    return bars