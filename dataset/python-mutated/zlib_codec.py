"""Python 'zlib_codec' Codec - zlib compression encoding.

This codec de/encodes from bytes to bytes.

Written by Marc-Andre Lemburg (mal@lemburg.com).
"""
import codecs
import zlib

def zlib_encode(input, errors='strict'):
    if False:
        i = 10
        return i + 15
    assert errors == 'strict'
    return (zlib.compress(input), len(input))

def zlib_decode(input, errors='strict'):
    if False:
        print('Hello World!')
    assert errors == 'strict'
    return (zlib.decompress(input), len(input))

class Codec(codecs.Codec):

    def encode(self, input, errors='strict'):
        if False:
            return 10
        return zlib_encode(input, errors)

    def decode(self, input, errors='strict'):
        if False:
            for i in range(10):
                print('nop')
        return zlib_decode(input, errors)

class IncrementalEncoder(codecs.IncrementalEncoder):

    def __init__(self, errors='strict'):
        if False:
            while True:
                i = 10
        assert errors == 'strict'
        self.errors = errors
        self.compressobj = zlib.compressobj()

    def encode(self, input, final=False):
        if False:
            i = 10
            return i + 15
        if final:
            c = self.compressobj.compress(input)
            return c + self.compressobj.flush()
        else:
            return self.compressobj.compress(input)

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.compressobj = zlib.compressobj()

class IncrementalDecoder(codecs.IncrementalDecoder):

    def __init__(self, errors='strict'):
        if False:
            while True:
                i = 10
        assert errors == 'strict'
        self.errors = errors
        self.decompressobj = zlib.decompressobj()

    def decode(self, input, final=False):
        if False:
            print('Hello World!')
        if final:
            c = self.decompressobj.decompress(input)
            return c + self.decompressobj.flush()
        else:
            return self.decompressobj.decompress(input)

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.decompressobj = zlib.decompressobj()

class StreamWriter(Codec, codecs.StreamWriter):
    charbuffertype = bytes

class StreamReader(Codec, codecs.StreamReader):
    charbuffertype = bytes

def getregentry():
    if False:
        for i in range(10):
            print('nop')
    return codecs.CodecInfo(name='zlib', encode=zlib_encode, decode=zlib_decode, incrementalencoder=IncrementalEncoder, incrementaldecoder=IncrementalDecoder, streamreader=StreamReader, streamwriter=StreamWriter, _is_text_encoding=False)