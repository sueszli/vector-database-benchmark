""" Python 'unicode-escape' Codec


Written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright CNRI, All Rights Reserved. NO WARRANTY.

"""
import codecs

class Codec(codecs.Codec):
    encode = codecs.unicode_escape_encode
    decode = codecs.unicode_escape_decode

class IncrementalEncoder(codecs.IncrementalEncoder):

    def encode(self, input, final=False):
        if False:
            print('Hello World!')
        return codecs.unicode_escape_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):

    def _buffer_decode(self, input, errors, final):
        if False:
            return 10
        return codecs.unicode_escape_decode(input, errors, final)

class StreamWriter(Codec, codecs.StreamWriter):
    pass

class StreamReader(Codec, codecs.StreamReader):

    def decode(self, input, errors='strict'):
        if False:
            return 10
        return codecs.unicode_escape_decode(input, errors, False)

def getregentry():
    if False:
        i = 10
        return i + 15
    return codecs.CodecInfo(name='unicode-escape', encode=Codec.encode, decode=Codec.decode, incrementalencoder=IncrementalEncoder, incrementaldecoder=IncrementalDecoder, streamwriter=StreamWriter, streamreader=StreamReader)