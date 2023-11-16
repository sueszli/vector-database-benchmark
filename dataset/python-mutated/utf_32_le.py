"""
Python 'utf-32-le' Codec
"""
import codecs
encode = codecs.utf_32_le_encode

def decode(input, errors='strict'):
    if False:
        for i in range(10):
            print('nop')
    return codecs.utf_32_le_decode(input, errors, True)

class IncrementalEncoder(codecs.IncrementalEncoder):

    def encode(self, input, final=False):
        if False:
            print('Hello World!')
        return codecs.utf_32_le_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    _buffer_decode = codecs.utf_32_le_decode

class StreamWriter(codecs.StreamWriter):
    encode = codecs.utf_32_le_encode

class StreamReader(codecs.StreamReader):
    decode = codecs.utf_32_le_decode

def getregentry():
    if False:
        i = 10
        return i + 15
    return codecs.CodecInfo(name='utf-32-le', encode=encode, decode=decode, incrementalencoder=IncrementalEncoder, incrementaldecoder=IncrementalDecoder, streamreader=StreamReader, streamwriter=StreamWriter)