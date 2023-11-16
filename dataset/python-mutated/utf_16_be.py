""" Python 'utf-16-be' Codec


Written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright CNRI, All Rights Reserved. NO WARRANTY.

"""
import codecs
encode = codecs.utf_16_be_encode

def decode(input, errors='strict'):
    if False:
        for i in range(10):
            print('nop')
    return codecs.utf_16_be_decode(input, errors, True)

class IncrementalEncoder(codecs.IncrementalEncoder):

    def encode(self, input, final=False):
        if False:
            while True:
                i = 10
        return codecs.utf_16_be_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    _buffer_decode = codecs.utf_16_be_decode

class StreamWriter(codecs.StreamWriter):
    encode = codecs.utf_16_be_encode

class StreamReader(codecs.StreamReader):
    decode = codecs.utf_16_be_decode

def getregentry():
    if False:
        i = 10
        return i + 15
    return codecs.CodecInfo(name='utf-16-be', encode=encode, decode=decode, incrementalencoder=IncrementalEncoder, incrementaldecoder=IncrementalDecoder, streamreader=StreamReader, streamwriter=StreamWriter)