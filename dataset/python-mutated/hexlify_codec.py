"""Python 'hex' Codec - 2-digit hex with spaces content transfer encoding.

Encode and decode may be a bit misleading at first sight...

The textual representation is a hex dump: e.g. "40 41"
The "encoded" data of this is the binary form, e.g. b"@A"

Therefore decoding is binary to text and thus converting binary data to hex dump.

"""
from __future__ import absolute_import
import codecs
import serial
try:
    unicode
except (NameError, AttributeError):
    unicode = str
HEXDIGITS = '0123456789ABCDEF'

def hex_encode(data, errors='strict'):
    if False:
        while True:
            i = 10
    "'40 41 42' -> b'@ab'"
    return (serial.to_bytes([int(h, 16) for h in data.split()]), len(data))

def hex_decode(data, errors='strict'):
    if False:
        print('Hello World!')
    "b'@ab' -> '40 41 42'"
    return (unicode(''.join(('{:02X} '.format(ord(b)) for b in serial.iterbytes(data)))), len(data))

class Codec(codecs.Codec):

    def encode(self, data, errors='strict'):
        if False:
            return 10
        "'40 41 42' -> b'@ab'"
        return serial.to_bytes([int(h, 16) for h in data.split()])

    def decode(self, data, errors='strict'):
        if False:
            while True:
                i = 10
        "b'@ab' -> '40 41 42'"
        return unicode(''.join(('{:02X} '.format(ord(b)) for b in serial.iterbytes(data))))

class IncrementalEncoder(codecs.IncrementalEncoder):
    """Incremental hex encoder"""

    def __init__(self, errors='strict'):
        if False:
            return 10
        self.errors = errors
        self.state = 0

    def reset(self):
        if False:
            return 10
        self.state = 0

    def getstate(self):
        if False:
            for i in range(10):
                print('nop')
        return self.state

    def setstate(self, state):
        if False:
            print('Hello World!')
        self.state = state

    def encode(self, data, final=False):
        if False:
            return 10
        "        Incremental encode, keep track of digits and emit a byte when a pair\n        of hex digits is found. The space is optional unless the error\n        handling is defined to be 'strict'.\n        "
        state = self.state
        encoded = []
        for c in data.upper():
            if c in HEXDIGITS:
                z = HEXDIGITS.index(c)
                if state:
                    encoded.append(z + (state & 240))
                    state = 0
                else:
                    state = 256 + (z << 4)
            elif c == ' ':
                if state and self.errors == 'strict':
                    raise UnicodeError('odd number of hex digits')
                state = 0
            elif self.errors == 'strict':
                raise UnicodeError('non-hex digit found: {!r}'.format(c))
        self.state = state
        return serial.to_bytes(encoded)

class IncrementalDecoder(codecs.IncrementalDecoder):
    """Incremental decoder"""

    def decode(self, data, final=False):
        if False:
            return 10
        return unicode(''.join(('{:02X} '.format(ord(b)) for b in serial.iterbytes(data))))

class StreamWriter(Codec, codecs.StreamWriter):
    """Combination of hexlify codec and StreamWriter"""

class StreamReader(Codec, codecs.StreamReader):
    """Combination of hexlify codec and StreamReader"""

def getregentry():
    if False:
        i = 10
        return i + 15
    'encodings module API'
    return codecs.CodecInfo(name='hexlify', encode=hex_encode, decode=hex_decode, incrementalencoder=IncrementalEncoder, incrementaldecoder=IncrementalDecoder, streamwriter=StreamWriter, streamreader=StreamReader)