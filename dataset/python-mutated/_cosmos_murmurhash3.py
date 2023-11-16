from ._cosmos_integers import UInt128, UInt64

def rotate_left_64(val, shift):
    if False:
        i = 10
        return i + 15
    return val << shift | val >> 64 - shift

def mix(value):
    if False:
        print('Hello World!')
    value ^= value >> 33
    value *= 18397679294719823053
    value = value & 18446744073709551615
    value ^= value >> 33
    value *= 14181476777654086739
    value = value & 18446744073709551615
    value ^= value >> 33
    return value

def murmurhash3_128(span: bytearray, seed: UInt128) -> UInt128:
    if False:
        i = 10
        return i + 15
    '\n    Python implementation of 128 bit murmurhash3 from Dot Net SDK. To match with other SDKs, It is recommended to\n    do the following with number values, especially floats as other SDKs use Doubles\n    -> bytearray(struct.pack("d", #)) where # represents any number. The d will treat it as a double.\n\n    :param bytearray span:\n        bytearray of value to hash\n    :param UInt128 seed:\n        seed value for murmurhash3, takes in a UInt128 value from Cosmos Integers\n    :return:\n        The hash value as a UInt128\n    :rtype:\n        UInt128'
    c1 = UInt64(9782798678568883157)
    c2 = UInt64(5545529020109919103)
    h1 = UInt64(seed.get_low())
    h2 = UInt64(seed.get_high())
    position = 0
    while position < len(span) - 15:
        k1 = UInt64(int.from_bytes(span[position:position + 8], 'little'))
        k2 = UInt64(int.from_bytes(span[position + 8:position + 16], 'little'))
        k1 *= c1
        k1.value = rotate_left_64(k1.value, 31)
        k1 *= c2
        h1 ^= k1
        h1.value = rotate_left_64(h1.value, 27)
        h1 += h2
        h1 = h1 * 5 + UInt64(1390208809)
        k2 *= c2
        k2.value = rotate_left_64(k2.value, 33)
        k2 *= c1
        h2 ^= k2
        h2.value = rotate_left_64(h2.value, 31)
        h2 += h1
        h2 = h2 * 5 + UInt64(944331445)
        position += 16
    k1 = UInt64(0)
    k2 = UInt64(0)
    n = len(span) & 15
    if n >= 15:
        k2 ^= UInt64(span[position + 14] << 48)
    if n >= 14:
        k2 ^= UInt64(span[position + 13] << 40)
    if n >= 13:
        k2 ^= UInt64(span[position + 12] << 32)
    if n >= 12:
        k2 ^= UInt64(span[position + 11] << 24)
    if n >= 11:
        k2 ^= UInt64(span[position + 10] << 16)
    if n >= 10:
        k2 ^= UInt64(span[position + 9] << 8)
    if n >= 9:
        k2 ^= UInt64(span[position + 8] << 0)
    k2 *= c2
    k2.value = rotate_left_64(k2.value, 33)
    k2 *= c1
    h2 ^= k2
    if n >= 8:
        k1 ^= UInt64(span[position + 7] << 56)
    if n >= 7:
        k1 ^= UInt64(span[position + 6] << 48)
    if n >= 6:
        k1 ^= UInt64(span[position + 5] << 40)
    if n >= 5:
        k1 ^= UInt64(span[position + 4] << 32)
    if n >= 4:
        k1 ^= UInt64(span[position + 3] << 24)
    if n >= 3:
        k1 ^= UInt64(span[position + 2] << 16)
    if n >= 2:
        k1 ^= UInt64(span[position + 1] << 8)
    if n >= 1:
        k1 ^= UInt64(span[position + 0] << 0)
    k1 *= c1
    k1.value = rotate_left_64(k1.value, 31)
    k1 *= c2
    h1 ^= k1
    h1 ^= UInt64(len(span))
    h2 ^= UInt64(len(span))
    h1 += h2
    h2 += h1
    h1 = mix(h1)
    h2 = mix(h2)
    h1 += h2
    h2 += h1
    return UInt128.create(int(h1.value), int(h2.value))