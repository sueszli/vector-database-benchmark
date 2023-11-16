import numpy as np
from numba import uint64, uint32, uint16, uint8
from numba.core.extending import register_jitable
from numba.np.random._constants import UINT32_MAX, UINT64_MAX, UINT16_MAX, UINT8_MAX
from numba.np.random.generator_core import next_uint32, next_uint64

@register_jitable
def gen_mask(max):
    if False:
        i = 10
        return i + 15
    mask = uint64(max)
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16
    mask |= mask >> 32
    return mask

@register_jitable
def buffered_bounded_bool(bitgen, off, rng, bcnt, buf):
    if False:
        while True:
            i = 10
    if rng == 0:
        return (off, bcnt, buf)
    if not bcnt:
        buf = next_uint32(bitgen)
        bcnt = 31
    else:
        buf >>= 1
        bcnt -= 1
    return (buf & 1 != 0, bcnt, buf)

@register_jitable
def buffered_uint8(bitgen, bcnt, buf):
    if False:
        print('Hello World!')
    if not bcnt:
        buf = next_uint32(bitgen)
        bcnt = 3
    else:
        buf >>= 8
        bcnt -= 1
    return (uint8(buf), bcnt, buf)

@register_jitable
def buffered_uint16(bitgen, bcnt, buf):
    if False:
        return 10
    if not bcnt:
        buf = next_uint32(bitgen)
        bcnt = 1
    else:
        buf >>= 16
        bcnt -= 1
    return (uint16(buf), bcnt, buf)

@register_jitable
def buffered_bounded_lemire_uint8(bitgen, rng, bcnt, buf):
    if False:
        print('Hello World!')
    "\n    Generates a random unsigned 8 bit integer bounded\n    within a given interval using Lemire's rejection.\n\n    The buffer acts as storage for a 32 bit integer\n    drawn from the associated BitGenerator so that\n    multiple integers of smaller bitsize can be generated\n    from a single draw of the BitGenerator.\n    "
    rng_excl = uint8(rng) + uint8(1)
    assert rng != 255
    (n, bcnt, buf) = buffered_uint8(bitgen, bcnt, buf)
    m = uint16(n * rng_excl)
    leftover = m & 255
    if leftover < rng_excl:
        threshold = (uint8(UINT8_MAX) - rng) % rng_excl
        while leftover < threshold:
            (n, bcnt, buf) = buffered_uint8(bitgen, bcnt, buf)
            m = uint16(n * rng_excl)
            leftover = m & 255
    return (m >> 8, bcnt, buf)

@register_jitable
def buffered_bounded_lemire_uint16(bitgen, rng, bcnt, buf):
    if False:
        print('Hello World!')
    "\n    Generates a random unsigned 16 bit integer bounded\n    within a given interval using Lemire's rejection.\n\n    The buffer acts as storage for a 32 bit integer\n    drawn from the associated BitGenerator so that\n    multiple integers of smaller bitsize can be generated\n    from a single draw of the BitGenerator.\n    "
    rng_excl = uint16(rng) + uint16(1)
    assert rng != 65535
    (n, bcnt, buf) = buffered_uint16(bitgen, bcnt, buf)
    m = uint32(n * rng_excl)
    leftover = m & 65535
    if leftover < rng_excl:
        threshold = (uint16(UINT16_MAX) - rng) % rng_excl
        while leftover < threshold:
            (n, bcnt, buf) = buffered_uint16(bitgen, bcnt, buf)
            m = uint32(n * rng_excl)
            leftover = m & 65535
    return (m >> 16, bcnt, buf)

@register_jitable
def buffered_bounded_lemire_uint32(bitgen, rng):
    if False:
        while True:
            i = 10
    "\n    Generates a random unsigned 32 bit integer bounded\n    within a given interval using Lemire's rejection.\n    "
    rng_excl = uint32(rng) + uint32(1)
    assert rng != 4294967295
    m = uint64(next_uint32(bitgen)) * uint64(rng_excl)
    leftover = m & 4294967295
    if leftover < rng_excl:
        threshold = (UINT32_MAX - rng) % rng_excl
        while leftover < threshold:
            m = uint64(next_uint32(bitgen)) * uint64(rng_excl)
            leftover = m & 4294967295
    return m >> 32

@register_jitable
def bounded_lemire_uint64(bitgen, rng):
    if False:
        return 10
    "\n    Generates a random unsigned 64 bit integer bounded\n    within a given interval using Lemire's rejection.\n    "
    rng_excl = uint64(rng) + uint64(1)
    assert rng != 18446744073709551615
    x = next_uint64(bitgen)
    leftover = uint64(x) * uint64(rng_excl)
    if leftover < rng_excl:
        threshold = (UINT64_MAX - rng) % rng_excl
        while leftover < threshold:
            x = next_uint64(bitgen)
            leftover = uint64(x) * uint64(rng_excl)
    x0 = x & uint64(4294967295)
    x1 = x >> 32
    rng_excl0 = rng_excl & uint64(4294967295)
    rng_excl1 = rng_excl >> 32
    w0 = x0 * rng_excl0
    t = x1 * rng_excl0 + (w0 >> 32)
    w1 = t & uint64(4294967295)
    w2 = t >> 32
    w1 += x0 * rng_excl1
    m1 = x1 * rng_excl1 + w2 + (w1 >> 32)
    return m1

@register_jitable
def random_bounded_uint64_fill(bitgen, low, rng, size, dtype):
    if False:
        return 10
    '\n    Returns a new array of given size with 64 bit integers\n    bounded by given interval.\n    '
    out = np.empty(size, dtype=dtype)
    if rng == 0:
        for i in np.ndindex(size):
            out[i] = low
    elif rng <= 4294967295:
        if rng == 4294967295:
            for i in np.ndindex(size):
                out[i] = low + next_uint32(bitgen)
        else:
            for i in np.ndindex(size):
                out[i] = low + buffered_bounded_lemire_uint32(bitgen, rng)
    elif rng == 18446744073709551615:
        for i in np.ndindex(size):
            out[i] = low + next_uint64(bitgen)
    else:
        for i in np.ndindex(size):
            out[i] = low + bounded_lemire_uint64(bitgen, rng)
    return out

@register_jitable
def random_bounded_uint32_fill(bitgen, low, rng, size, dtype):
    if False:
        while True:
            i = 10
    '\n    Returns a new array of given size with 32 bit integers\n    bounded by given interval.\n    '
    out = np.empty(size, dtype=dtype)
    if rng == 0:
        for i in np.ndindex(size):
            out[i] = low
    elif rng == 4294967295:
        for i in np.ndindex(size):
            out[i] = low + next_uint32(bitgen)
    else:
        for i in np.ndindex(size):
            out[i] = low + buffered_bounded_lemire_uint32(bitgen, rng)
    return out

@register_jitable
def random_bounded_uint16_fill(bitgen, low, rng, size, dtype):
    if False:
        i = 10
        return i + 15
    '\n    Returns a new array of given size with 16 bit integers\n    bounded by given interval.\n    '
    buf = 0
    bcnt = 0
    out = np.empty(size, dtype=dtype)
    if rng == 0:
        for i in np.ndindex(size):
            out[i] = low
    elif rng == 65535:
        for i in np.ndindex(size):
            (val, bcnt, buf) = buffered_uint16(bitgen, bcnt, buf)
            out[i] = low + val
    else:
        for i in np.ndindex(size):
            (val, bcnt, buf) = buffered_bounded_lemire_uint16(bitgen, rng, bcnt, buf)
            out[i] = low + val
    return out

@register_jitable
def random_bounded_uint8_fill(bitgen, low, rng, size, dtype):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a new array of given size with 8 bit integers\n    bounded by given interval.\n    '
    buf = 0
    bcnt = 0
    out = np.empty(size, dtype=dtype)
    if rng == 0:
        for i in np.ndindex(size):
            out[i] = low
    elif rng == 255:
        for i in np.ndindex(size):
            (val, bcnt, buf) = buffered_uint8(bitgen, bcnt, buf)
            out[i] = low + val
    else:
        for i in np.ndindex(size):
            (val, bcnt, buf) = buffered_bounded_lemire_uint8(bitgen, rng, bcnt, buf)
            out[i] = low + val
    return out

@register_jitable
def random_bounded_bool_fill(bitgen, low, rng, size, dtype):
    if False:
        return 10
    '\n    Returns a new array of given size with boolean values.\n    '
    buf = 0
    bcnt = 0
    out = np.empty(size, dtype=dtype)
    for i in np.ndindex(size):
        (val, bcnt, buf) = buffered_bounded_bool(bitgen, low, rng, bcnt, buf)
        out[i] = low + val
    return out

@register_jitable
def _randint_arg_check(low, high, endpoint, lower_bound, upper_bound):
    if False:
        return 10
    '\n    Check that low and high are within the bounds\n    for the given datatype.\n    '
    if low < lower_bound:
        raise ValueError('low is out of bounds')
    if high > 0:
        high = uint64(high)
        if not endpoint:
            high -= uint64(1)
        upper_bound = uint64(upper_bound)
        if low > 0:
            low = uint64(low)
        if high > upper_bound:
            raise ValueError('high is out of bounds')
        if low > high:
            raise ValueError('low is greater than high in given interval')
    else:
        if high > upper_bound:
            raise ValueError('high is out of bounds')
        if low > high:
            raise ValueError('low is greater than high in given interval')

@register_jitable
def random_interval(bitgen, max_val):
    if False:
        print('Hello World!')
    if max_val == 0:
        return 0
    max_val = uint64(max_val)
    mask = uint64(gen_mask(max_val))
    if max_val <= 4294967295:
        value = uint64(next_uint32(bitgen)) & mask
        while value > max_val:
            value = uint64(next_uint32(bitgen)) & mask
    else:
        value = next_uint64(bitgen) & mask
        while value > max_val:
            value = next_uint64(bitgen) & mask
    return uint64(value)