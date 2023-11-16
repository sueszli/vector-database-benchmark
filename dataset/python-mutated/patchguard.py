import struct

def rol(value, count):
    if False:
        while True:
            i = 10
    'A rotate-left instruction in Python'
    for y in range(count):
        value *= 2
        if value > 18446744073709551615:
            value -= 18446744073709551616
            value += 1
    return value

def bswap(value):
    if False:
        while True:
            i = 10
    'A byte-swap instruction in Python'
    (hi, lo) = struct.unpack('>II', struct.pack('<Q', value))
    return hi << 32 | lo