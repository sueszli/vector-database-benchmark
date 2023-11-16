from __future__ import print_function
import pyperf
from kafka.vendor import six
test_data = [(b'\x00', 0), (b'\x01', -1), (b'\x02', 1), (b'~', 63), (b'\x7f', -64), (b'\x80\x01', 64), (b'\x81\x01', -65), (b'\xfe\x7f', 8191), (b'\xff\x7f', -8192), (b'\x80\x80\x01', 8192), (b'\x81\x80\x01', -8193), (b'\xfe\xff\x7f', 1048575), (b'\xff\xff\x7f', -1048576), (b'\x80\x80\x80\x01', 1048576), (b'\x81\x80\x80\x01', -1048577), (b'\xfe\xff\xff\x7f', 134217727), (b'\xff\xff\xff\x7f', -134217728), (b'\x80\x80\x80\x80\x01', 134217728), (b'\x81\x80\x80\x80\x01', -134217729), (b'\xfe\xff\xff\xff\x7f', 17179869183), (b'\xff\xff\xff\xff\x7f', -17179869184), (b'\x80\x80\x80\x80\x80\x01', 17179869184), (b'\x81\x80\x80\x80\x80\x01', -17179869185), (b'\xfe\xff\xff\xff\xff\x7f', 2199023255551), (b'\xff\xff\xff\xff\xff\x7f', -2199023255552), (b'\x80\x80\x80\x80\x80\x80\x01', 2199023255552), (b'\x81\x80\x80\x80\x80\x80\x01', -2199023255553), (b'\xfe\xff\xff\xff\xff\xff\x7f', 281474976710655), (b'\xff\xff\xff\xff\xff\xff\x7f', -281474976710656), (b'\x80\x80\x80\x80\x80\x80\x80\x01', 281474976710656), (b'\x81\x80\x80\x80\x80\x80\x80\x01', -281474976710657), (b'\xfe\xff\xff\xff\xff\xff\xff\x7f', 36028797018963967), (b'\xff\xff\xff\xff\xff\xff\xff\x7f', -36028797018963968), (b'\x80\x80\x80\x80\x80\x80\x80\x80\x01', 36028797018963968), (b'\x81\x80\x80\x80\x80\x80\x80\x80\x01', -36028797018963969), (b'\xfe\xff\xff\xff\xff\xff\xff\xff\x7f', 4611686018427387903), (b'\xff\xff\xff\xff\xff\xff\xff\xff\x7f', -4611686018427387904), (b'\x80\x80\x80\x80\x80\x80\x80\x80\x80\x01', 4611686018427387904), (b'\x81\x80\x80\x80\x80\x80\x80\x80\x80\x01', -4611686018427387905)]
BENCH_VALUES_ENC = [60, -8192, 1048575, 134217727, -17179869184, 2199023255551]
BENCH_VALUES_DEC = [b'~', b'\xff\x7f', b'\xfe\xff\x7f', b'\xff\xff\xff\x7f', b'\x80\x80\x80\x80\x01', b'\xfe\xff\xff\xff\xff\x7f']
BENCH_VALUES_DEC = list(map(bytearray, BENCH_VALUES_DEC))

def _assert_valid_enc(enc_func):
    if False:
        return 10
    for (encoded, decoded) in test_data:
        assert enc_func(decoded) == encoded, decoded

def _assert_valid_dec(dec_func):
    if False:
        print('Hello World!')
    for (encoded, decoded) in test_data:
        (res, pos) = dec_func(bytearray(encoded))
        assert res == decoded, (decoded, res)
        assert pos == len(encoded), (decoded, pos)

def _assert_valid_size(size_func):
    if False:
        return 10
    for (encoded, decoded) in test_data:
        assert size_func(decoded) == len(encoded), decoded

def encode_varint_1(num):
    if False:
        return 10
    ' Encode an integer to a varint presentation. See\n    https://developers.google.com/protocol-buffers/docs/encoding?csw=1#varints\n    on how those can be produced.\n\n        Arguments:\n            num (int): Value to encode\n\n        Returns:\n            bytearray: Encoded presentation of integer with length from 1 to 10\n                 bytes\n    '
    num = num << 1 ^ num >> 63
    buf = bytearray(10)
    for i in range(10):
        buf[i] = num & 127 | (128 if num > 127 else 0)
        num = num >> 7
        if num == 0:
            break
    else:
        raise ValueError('Out of double range')
    return buf[:i + 1]
_assert_valid_enc(encode_varint_1)

def encode_varint_2(value, int2byte=six.int2byte):
    if False:
        i = 10
        return i + 15
    value = value << 1 ^ value >> 63
    bits = value & 127
    value >>= 7
    res = b''
    while value:
        res += int2byte(128 | bits)
        bits = value & 127
        value >>= 7
    return res + int2byte(bits)
_assert_valid_enc(encode_varint_2)

def encode_varint_3(value, buf):
    if False:
        return 10
    append = buf.append
    value = value << 1 ^ value >> 63
    bits = value & 127
    value >>= 7
    while value:
        append(128 | bits)
        bits = value & 127
        value >>= 7
    append(bits)
    return value
for (encoded, decoded) in test_data:
    res = bytearray()
    encode_varint_3(decoded, res)
    assert res == encoded

def encode_varint_4(value, int2byte=six.int2byte):
    if False:
        return 10
    value = value << 1 ^ value >> 63
    if value <= 127:
        return int2byte(value)
    if value <= 16383:
        return int2byte(128 | value & 127) + int2byte(value >> 7)
    if value <= 2097151:
        return int2byte(128 | value & 127) + int2byte(128 | value >> 7 & 127) + int2byte(value >> 14)
    if value <= 268435455:
        return int2byte(128 | value & 127) + int2byte(128 | value >> 7 & 127) + int2byte(128 | value >> 14 & 127) + int2byte(value >> 21)
    if value <= 34359738367:
        return int2byte(128 | value & 127) + int2byte(128 | value >> 7 & 127) + int2byte(128 | value >> 14 & 127) + int2byte(128 | value >> 21 & 127) + int2byte(value >> 28)
    else:
        bits = value & 127
        value >>= 7
        res = b''
        while value:
            res += int2byte(128 | bits)
            bits = value & 127
            value >>= 7
        return res + int2byte(bits)
_assert_valid_enc(encode_varint_4)

def encode_varint_5(value, buf, pos=0):
    if False:
        return 10
    value = value << 1 ^ value >> 63
    bits = value & 127
    value >>= 7
    while value:
        buf[pos] = 128 | bits
        bits = value & 127
        value >>= 7
        pos += 1
    buf[pos] = bits
    return pos + 1
for (encoded, decoded) in test_data:
    res = bytearray(10)
    written = encode_varint_5(decoded, res)
    assert res[:written] == encoded

def encode_varint_6(value, buf):
    if False:
        for i in range(10):
            print('nop')
    append = buf.append
    value = value << 1 ^ value >> 63
    if value <= 127:
        append(value)
        return 1
    if value <= 16383:
        append(128 | value & 127)
        append(value >> 7)
        return 2
    if value <= 2097151:
        append(128 | value & 127)
        append(128 | value >> 7 & 127)
        append(value >> 14)
        return 3
    if value <= 268435455:
        append(128 | value & 127)
        append(128 | value >> 7 & 127)
        append(128 | value >> 14 & 127)
        append(value >> 21)
        return 4
    if value <= 34359738367:
        append(128 | value & 127)
        append(128 | value >> 7 & 127)
        append(128 | value >> 14 & 127)
        append(128 | value >> 21 & 127)
        append(value >> 28)
        return 5
    else:
        bits = value & 127
        value >>= 7
        i = 0
        while value:
            append(128 | bits)
            bits = value & 127
            value >>= 7
            i += 1
    append(bits)
    return i
for (encoded, decoded) in test_data:
    res = bytearray()
    encode_varint_6(decoded, res)
    assert res == encoded

def size_of_varint_1(value):
    if False:
        i = 10
        return i + 15
    ' Number of bytes needed to encode an integer in variable-length format.\n    '
    value = value << 1 ^ value >> 63
    res = 0
    while True:
        res += 1
        value = value >> 7
        if value == 0:
            break
    return res
_assert_valid_size(size_of_varint_1)

def size_of_varint_2(value):
    if False:
        while True:
            i = 10
    ' Number of bytes needed to encode an integer in variable-length format.\n    '
    value = value << 1 ^ value >> 63
    if value <= 127:
        return 1
    if value <= 16383:
        return 2
    if value <= 2097151:
        return 3
    if value <= 268435455:
        return 4
    if value <= 34359738367:
        return 5
    if value <= 4398046511103:
        return 6
    if value <= 562949953421311:
        return 7
    if value <= 72057594037927935:
        return 8
    if value <= 9223372036854775807:
        return 9
    return 10
_assert_valid_size(size_of_varint_2)
if six.PY3:

    def _read_byte(memview, pos):
        if False:
            for i in range(10):
                print('nop')
        ' Read a byte from memoryview as an integer\n\n            Raises:\n                IndexError: if position is out of bounds\n        '
        return memview[pos]
else:

    def _read_byte(memview, pos):
        if False:
            for i in range(10):
                print('nop')
        ' Read a byte from memoryview as an integer\n\n            Raises:\n                IndexError: if position is out of bounds\n        '
        return ord(memview[pos])

def decode_varint_1(buffer, pos=0):
    if False:
        while True:
            i = 10
    ' Decode an integer from a varint presentation. See\n    https://developers.google.com/protocol-buffers/docs/encoding?csw=1#varints\n    on how those can be produced.\n\n        Arguments:\n            buffer (bytes-like): any object acceptable by ``memoryview``\n            pos (int): optional position to read from\n\n        Returns:\n            (int, int): Decoded int value and next read position\n    '
    value = 0
    shift = 0
    memview = memoryview(buffer)
    for i in range(pos, pos + 10):
        try:
            byte = _read_byte(memview, i)
        except IndexError:
            raise ValueError('End of byte stream')
        if byte & 128 != 0:
            value |= (byte & 127) << shift
            shift += 7
        else:
            value |= byte << shift
            break
    else:
        raise ValueError('Out of double range')
    return (value >> 1 ^ -(value & 1), i + 1)
_assert_valid_dec(decode_varint_1)

def decode_varint_2(buffer, pos=0):
    if False:
        while True:
            i = 10
    result = 0
    shift = 0
    while 1:
        b = buffer[pos]
        result |= (b & 127) << shift
        pos += 1
        if not b & 128:
            return (result >> 1 ^ -(result & 1), pos)
        shift += 7
        if shift >= 64:
            raise ValueError('Out of int64 range')
_assert_valid_dec(decode_varint_2)

def decode_varint_3(buffer, pos=0):
    if False:
        i = 10
        return i + 15
    result = buffer[pos]
    if not result & 129:
        return (result >> 1, pos + 1)
    if not result & 128:
        return (result >> 1 ^ ~0, pos + 1)
    result &= 127
    pos += 1
    shift = 7
    while 1:
        b = buffer[pos]
        result |= (b & 127) << shift
        pos += 1
        if not b & 128:
            return (result >> 1 ^ -(result & 1), pos)
        shift += 7
        if shift >= 64:
            raise ValueError('Out of int64 range')
_assert_valid_dec(decode_varint_3)
runner = pyperf.Runner()
for bench_func in [encode_varint_1, encode_varint_2, encode_varint_4]:
    for (i, value) in enumerate(BENCH_VALUES_ENC):
        runner.bench_func('{}_{}byte'.format(bench_func.__name__, i + 1), bench_func, value)
for bench_func in [encode_varint_3, encode_varint_5, encode_varint_6]:
    for (i, value) in enumerate(BENCH_VALUES_ENC):
        fname = bench_func.__name__
        runner.timeit('{}_{}byte'.format(fname, i + 1), stmt='{}({}, buffer)'.format(fname, value), setup='from __main__ import {}; buffer = bytearray(10)'.format(fname))
for bench_func in [size_of_varint_1, size_of_varint_2]:
    for (i, value) in enumerate(BENCH_VALUES_ENC):
        runner.bench_func('{}_{}byte'.format(bench_func.__name__, i + 1), bench_func, value)
for bench_func in [decode_varint_1, decode_varint_2, decode_varint_3]:
    for (i, value) in enumerate(BENCH_VALUES_DEC):
        runner.bench_func('{}_{}byte'.format(bench_func.__name__, i + 1), bench_func, value)