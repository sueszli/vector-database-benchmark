"""Codec for quoted-printable encoding.

This codec de/encodes from bytes to bytes.
"""
import codecs
import quopri
from io import BytesIO

def quopri_encode(input, errors='strict'):
    if False:
        print('Hello World!')
    assert errors == 'strict'
    f = BytesIO(input)
    g = BytesIO()
    quopri.encode(f, g, quotetabs=True)
    return (g.getvalue(), len(input))

def quopri_decode(input, errors='strict'):
    if False:
        return 10
    assert errors == 'strict'
    f = BytesIO(input)
    g = BytesIO()
    quopri.decode(f, g)
    return (g.getvalue(), len(input))

class Codec(codecs.Codec):

    def encode(self, input, errors='strict'):
        if False:
            return 10
        return quopri_encode(input, errors)

    def decode(self, input, errors='strict'):
        if False:
            i = 10
            return i + 15
        return quopri_decode(input, errors)

class IncrementalEncoder(codecs.IncrementalEncoder):

    def encode(self, input, final=False):
        if False:
            while True:
                i = 10
        return quopri_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):

    def decode(self, input, final=False):
        if False:
            i = 10
            return i + 15
        return quopri_decode(input, self.errors)[0]

class StreamWriter(Codec, codecs.StreamWriter):
    charbuffertype = bytes

class StreamReader(Codec, codecs.StreamReader):
    charbuffertype = bytes

def getregentry():
    if False:
        return 10
    return codecs.CodecInfo(name='quopri', encode=quopri_encode, decode=quopri_decode, incrementalencoder=IncrementalEncoder, incrementaldecoder=IncrementalDecoder, streamwriter=StreamWriter, streamreader=StreamReader, _is_text_encoding=False)