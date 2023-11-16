"""
Helper functions for np.timedelta64 and np.datetime64.
For now, multiples-of-units (for example timedeltas expressed in tens
of seconds) are not supported.
"""
import numpy as np
DATETIME_UNITS = {'Y': 0, 'M': 1, 'W': 2, 'D': 4, 'h': 5, 'm': 6, 's': 7, 'ms': 8, 'us': 9, 'ns': 10, 'ps': 11, 'fs': 12, 'as': 13, '': 14}
NAT = np.timedelta64('nat').astype(np.int64)

def same_kind(src, dest):
    if False:
        while True:
            i = 10
    '\n    Whether the *src* and *dest* units are of the same kind.\n    '
    return (DATETIME_UNITS[src] < 5) == (DATETIME_UNITS[dest] < 5)

def can_cast_timedelta_units(src, dest):
    if False:
        while True:
            i = 10
    src = DATETIME_UNITS[src]
    dest = DATETIME_UNITS[dest]
    if src == dest:
        return True
    if src == 14:
        return True
    if src > dest:
        return False
    if dest == 14:
        return False
    if src <= 1 and dest > 1:
        return False
    return True
_factors = {0: (1, 12), 2: (4, 7), 4: (5, 24), 5: (6, 60), 6: (7, 60), 7: (8, 1000), 8: (9, 1000), 9: (10, 1000), 10: (11, 1000), 11: (12, 1000), 12: (13, 1000)}

def _get_conversion_multiplier(big_unit_code, small_unit_code):
    if False:
        while True:
            i = 10
    '\n    Return an integer multiplier allowing to convert from *big_unit_code*\n    to *small_unit_code*.\n    None is returned if the conversion is not possible through a\n    simple integer multiplication.\n    '
    if big_unit_code == 14:
        return 1
    c = big_unit_code
    factor = 1
    while c < small_unit_code:
        try:
            (c, mult) = _factors[c]
        except KeyError:
            return None
        factor *= mult
    if c == small_unit_code:
        return factor
    else:
        return None

def get_timedelta_conversion_factor(src_unit, dest_unit):
    if False:
        while True:
            i = 10
    '\n    Return an integer multiplier allowing to convert from timedeltas\n    of *src_unit* to *dest_unit*.\n    '
    return _get_conversion_multiplier(DATETIME_UNITS[src_unit], DATETIME_UNITS[dest_unit])

def get_datetime_timedelta_conversion(datetime_unit, timedelta_unit):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute a possible conversion for combining *datetime_unit* and\n    *timedelta_unit* (presumably for adding or subtracting).\n    Return (result unit, integer datetime multiplier, integer timedelta\n    multiplier). RuntimeError is raised if the combination is impossible.\n    '
    dt_unit_code = DATETIME_UNITS[datetime_unit]
    td_unit_code = DATETIME_UNITS[timedelta_unit]
    if td_unit_code == 14 or dt_unit_code == 14:
        return (datetime_unit, 1, 1)
    if td_unit_code < 2 and dt_unit_code >= 2:
        raise RuntimeError('cannot combine datetime64(%r) and timedelta64(%r)' % (datetime_unit, timedelta_unit))
    (dt_factor, td_factor) = (1, 1)
    if dt_unit_code == 0:
        if td_unit_code >= 4:
            dt_factor = 97 + 400 * 365
            td_factor = 400
            dt_unit_code = 4
        elif td_unit_code == 2:
            dt_factor = 97 + 400 * 365
            td_factor = 400 * 7
            dt_unit_code = 2
    elif dt_unit_code == 1:
        if td_unit_code >= 4:
            dt_factor = 97 + 400 * 365
            td_factor = 400 * 12
            dt_unit_code = 4
        elif td_unit_code == 2:
            dt_factor = 97 + 400 * 365
            td_factor = 400 * 12 * 7
            dt_unit_code = 2
    if td_unit_code >= dt_unit_code:
        factor = _get_conversion_multiplier(dt_unit_code, td_unit_code)
        assert factor is not None, (dt_unit_code, td_unit_code)
        return (timedelta_unit, dt_factor * factor, td_factor)
    else:
        factor = _get_conversion_multiplier(td_unit_code, dt_unit_code)
        assert factor is not None, (dt_unit_code, td_unit_code)
        return (datetime_unit, dt_factor, td_factor * factor)

def combine_datetime_timedelta_units(datetime_unit, timedelta_unit):
    if False:
        print('Hello World!')
    '\n    Return the unit result of combining *datetime_unit* with *timedelta_unit*\n    (e.g. by adding or subtracting).  None is returned if combining\n    those units is forbidden.\n    '
    dt_unit_code = DATETIME_UNITS[datetime_unit]
    td_unit_code = DATETIME_UNITS[timedelta_unit]
    if dt_unit_code == 14:
        return timedelta_unit
    elif td_unit_code == 14:
        return datetime_unit
    if td_unit_code < 2 and dt_unit_code >= 2:
        return None
    if dt_unit_code > td_unit_code:
        return datetime_unit
    else:
        return timedelta_unit

def get_best_unit(unit_a, unit_b):
    if False:
        print('Hello World!')
    '\n    Get the best (i.e. finer-grained) of two units.\n    '
    a = DATETIME_UNITS[unit_a]
    b = DATETIME_UNITS[unit_b]
    if a == 14:
        return unit_b
    if b == 14:
        return unit_a
    if b > a:
        return unit_b
    return unit_a

def datetime_minimum(a, b):
    if False:
        return 10
    pass

def datetime_maximum(a, b):
    if False:
        print('Hello World!')
    pass