import math
import struct
from sys import float_info
from typing import Callable, Optional, SupportsFloat
STRUCT_FORMATS = {16: ('!H', '!e'), 32: ('!I', '!f'), 64: ('!Q', '!d')}

def reinterpret_bits(x, from_, to):
    if False:
        while True:
            i = 10
    return struct.unpack(to, struct.pack(from_, x))[0]

def float_of(x, width):
    if False:
        return 10
    assert width in (16, 32, 64)
    if width == 64:
        return float(x)
    elif width == 32:
        return reinterpret_bits(float(x), '!f', '!f')
    else:
        return reinterpret_bits(float(x), '!e', '!e')

def is_negative(x: SupportsFloat) -> bool:
    if False:
        for i in range(10):
            print('nop')
    try:
        return math.copysign(1.0, x) < 0
    except TypeError:
        raise TypeError(f'Expected float but got {x!r} of type {type(x).__name__}') from None

def count_between_floats(x, y, width=64):
    if False:
        for i in range(10):
            print('nop')
    assert x <= y
    if is_negative(x):
        if is_negative(y):
            return float_to_int(x, width) - float_to_int(y, width) + 1
        else:
            return count_between_floats(x, -0.0, width) + count_between_floats(0.0, y, width)
    else:
        assert not is_negative(y)
        return float_to_int(y, width) - float_to_int(x, width) + 1

def float_to_int(value, width=64):
    if False:
        i = 10
        return i + 15
    (fmt_int, fmt_flt) = STRUCT_FORMATS[width]
    return reinterpret_bits(value, fmt_flt, fmt_int)

def int_to_float(value, width=64):
    if False:
        i = 10
        return i + 15
    (fmt_int, fmt_flt) = STRUCT_FORMATS[width]
    return reinterpret_bits(value, fmt_int, fmt_flt)

def next_up(value, width=64):
    if False:
        print('Hello World!')
    "Return the first float larger than finite `val` - IEEE 754's `nextUp`.\n\n    From https://stackoverflow.com/a/10426033, with thanks to Mark Dickinson.\n    "
    assert isinstance(value, float), f'{value!r} of type {type(value)}'
    if math.isnan(value) or (math.isinf(value) and value > 0):
        return value
    if value == 0.0 and is_negative(value):
        return 0.0
    (fmt_int, fmt_flt) = STRUCT_FORMATS[width]
    fmt_int = fmt_int.lower()
    n = reinterpret_bits(value, fmt_flt, fmt_int)
    if n >= 0:
        n += 1
    else:
        n -= 1
    return reinterpret_bits(n, fmt_int, fmt_flt)

def next_down(value, width=64):
    if False:
        print('Hello World!')
    return -next_up(-value, width)

def next_down_normal(value, width, allow_subnormal):
    if False:
        for i in range(10):
            print('nop')
    value = next_down(value, width)
    if not allow_subnormal and 0 < abs(value) < width_smallest_normals[width]:
        return 0.0 if value > 0 else -width_smallest_normals[width]
    return value

def next_up_normal(value, width, allow_subnormal):
    if False:
        print('Hello World!')
    return -next_down_normal(-value, width, allow_subnormal)
width_smallest_normals = {16: 2 ** (-(2 ** (5 - 1) - 2)), 32: 2 ** (-(2 ** (8 - 1) - 2)), 64: 2 ** (-(2 ** (11 - 1) - 2))}
assert width_smallest_normals[64] == float_info.min

def make_float_clamper(min_float: float=0.0, max_float: float=math.inf, *, allow_zero: bool=False) -> Optional[Callable[[float], float]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a function that clamps positive floats into the given bounds.\n\n    Returns None when no values are allowed (min > max and zero is not allowed).\n    '
    if max_float < min_float:
        if allow_zero:
            min_float = max_float = 0.0
        else:
            return None
    range_size = min(max_float - min_float, float_info.max)
    mantissa_mask = (1 << 52) - 1

    def float_clamper(float_val: float) -> float:
        if False:
            return 10
        if min_float <= float_val <= max_float:
            return float_val
        if float_val == 0.0 and allow_zero:
            return float_val
        mant = float_to_int(float_val) & mantissa_mask
        float_val = min_float + range_size * (mant / mantissa_mask)
        return max(min_float, min(max_float, float_val))
    return float_clamper