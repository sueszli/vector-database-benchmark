""" Python 'oem' Codec for Windows

"""
from codecs import oem_encode, oem_decode
import codecs
encode = oem_encode

def decode(input, errors='strict'):
    if False:
        i = 10
        return i + 15
    return oem_decode(input, errors, True)

class IncrementalEncoder(codecs.IncrementalEncoder):

    def encode(self, input, final=False):
        if False:
            print('Hello World!')
        return oem_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    _buffer_decode = oem_decode

class StreamWriter(codecs.StreamWriter):
    encode = oem_encode

class StreamReader(codecs.StreamReader):
    decode = oem_decode

def getregentry():
    if False:
        while True:
            i = 10
    return codecs.CodecInfo(name='oem', encode=encode, decode=decode, incrementalencoder=IncrementalEncoder, incrementaldecoder=IncrementalDecoder, streamreader=StreamReader, streamwriter=StreamWriter)