"""Binary input/output support routines."""
from struct import pack, unpack_from

def i8(c):
    if False:
        print('Hello World!')
    return c if c.__class__ is int else c[0]

def o8(i):
    if False:
        print('Hello World!')
    return bytes((i & 255,))

def i16le(c, o=0):
    if False:
        i = 10
        return i + 15
    '\n    Converts a 2-bytes (16 bits) string to an unsigned integer.\n\n    :param c: string containing bytes to convert\n    :param o: offset of bytes to convert in string\n    '
    return unpack_from('<H', c, o)[0]

def si16le(c, o=0):
    if False:
        return 10
    '\n    Converts a 2-bytes (16 bits) string to a signed integer.\n\n    :param c: string containing bytes to convert\n    :param o: offset of bytes to convert in string\n    '
    return unpack_from('<h', c, o)[0]

def si16be(c, o=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts a 2-bytes (16 bits) string to a signed integer, big endian.\n\n    :param c: string containing bytes to convert\n    :param o: offset of bytes to convert in string\n    '
    return unpack_from('>h', c, o)[0]

def i32le(c, o=0):
    if False:
        while True:
            i = 10
    '\n    Converts a 4-bytes (32 bits) string to an unsigned integer.\n\n    :param c: string containing bytes to convert\n    :param o: offset of bytes to convert in string\n    '
    return unpack_from('<I', c, o)[0]

def si32le(c, o=0):
    if False:
        i = 10
        return i + 15
    '\n    Converts a 4-bytes (32 bits) string to a signed integer.\n\n    :param c: string containing bytes to convert\n    :param o: offset of bytes to convert in string\n    '
    return unpack_from('<i', c, o)[0]

def i16be(c, o=0):
    if False:
        for i in range(10):
            print('nop')
    return unpack_from('>H', c, o)[0]

def i32be(c, o=0):
    if False:
        while True:
            i = 10
    return unpack_from('>I', c, o)[0]

def o16le(i):
    if False:
        i = 10
        return i + 15
    return pack('<H', i)

def o32le(i):
    if False:
        for i in range(10):
            print('nop')
    return pack('<I', i)

def o16be(i):
    if False:
        i = 10
        return i + 15
    return pack('>H', i)

def o32be(i):
    if False:
        while True:
            i = 10
    return pack('>I', i)