""" Python 'utf-16' Codec


Written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright CNRI, All Rights Reserved. NO WARRANTY.

"""
import codecs, sys
encode = codecs.utf_16_encode

def decode(input, errors='strict'):
    if False:
        print('Hello World!')
    return codecs.utf_16_decode(input, errors, True)

class IncrementalEncoder(codecs.IncrementalEncoder):

    def __init__(self, errors='strict'):
        if False:
            i = 10
            return i + 15
        codecs.IncrementalEncoder.__init__(self, errors)
        self.encoder = None

    def encode(self, input, final=False):
        if False:
            print('Hello World!')
        if self.encoder is None:
            result = codecs.utf_16_encode(input, self.errors)[0]
            if sys.byteorder == 'little':
                self.encoder = codecs.utf_16_le_encode
            else:
                self.encoder = codecs.utf_16_be_encode
            return result
        return self.encoder(input, self.errors)[0]

    def reset(self):
        if False:
            print('Hello World!')
        codecs.IncrementalEncoder.reset(self)
        self.encoder = None

    def getstate(self):
        if False:
            while True:
                i = 10
        return 2 if self.encoder is None else 0

    def setstate(self, state):
        if False:
            i = 10
            return i + 15
        if state:
            self.encoder = None
        elif sys.byteorder == 'little':
            self.encoder = codecs.utf_16_le_encode
        else:
            self.encoder = codecs.utf_16_be_encode

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):

    def __init__(self, errors='strict'):
        if False:
            for i in range(10):
                print('nop')
        codecs.BufferedIncrementalDecoder.__init__(self, errors)
        self.decoder = None

    def _buffer_decode(self, input, errors, final):
        if False:
            while True:
                i = 10
        if self.decoder is None:
            (output, consumed, byteorder) = codecs.utf_16_ex_decode(input, errors, 0, final)
            if byteorder == -1:
                self.decoder = codecs.utf_16_le_decode
            elif byteorder == 1:
                self.decoder = codecs.utf_16_be_decode
            elif consumed >= 2:
                raise UnicodeError('UTF-16 stream does not start with BOM')
            return (output, consumed)
        return self.decoder(input, self.errors, final)

    def reset(self):
        if False:
            i = 10
            return i + 15
        codecs.BufferedIncrementalDecoder.reset(self)
        self.decoder = None

    def getstate(self):
        if False:
            while True:
                i = 10
        state = codecs.BufferedIncrementalDecoder.getstate(self)[0]
        if self.decoder is None:
            return (state, 2)
        addstate = int((sys.byteorder == 'big') != (self.decoder is codecs.utf_16_be_decode))
        return (state, addstate)

    def setstate(self, state):
        if False:
            print('Hello World!')
        codecs.BufferedIncrementalDecoder.setstate(self, state)
        state = state[1]
        if state == 0:
            self.decoder = codecs.utf_16_be_decode if sys.byteorder == 'big' else codecs.utf_16_le_decode
        elif state == 1:
            self.decoder = codecs.utf_16_le_decode if sys.byteorder == 'big' else codecs.utf_16_be_decode
        else:
            self.decoder = None

class StreamWriter(codecs.StreamWriter):

    def __init__(self, stream, errors='strict'):
        if False:
            return 10
        codecs.StreamWriter.__init__(self, stream, errors)
        self.encoder = None

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        codecs.StreamWriter.reset(self)
        self.encoder = None

    def encode(self, input, errors='strict'):
        if False:
            print('Hello World!')
        if self.encoder is None:
            result = codecs.utf_16_encode(input, errors)
            if sys.byteorder == 'little':
                self.encoder = codecs.utf_16_le_encode
            else:
                self.encoder = codecs.utf_16_be_encode
            return result
        else:
            return self.encoder(input, errors)

class StreamReader(codecs.StreamReader):

    def reset(self):
        if False:
            i = 10
            return i + 15
        codecs.StreamReader.reset(self)
        try:
            del self.decode
        except AttributeError:
            pass

    def decode(self, input, errors='strict'):
        if False:
            i = 10
            return i + 15
        (object, consumed, byteorder) = codecs.utf_16_ex_decode(input, errors, 0, False)
        if byteorder == -1:
            self.decode = codecs.utf_16_le_decode
        elif byteorder == 1:
            self.decode = codecs.utf_16_be_decode
        elif consumed >= 2:
            raise UnicodeError('UTF-16 stream does not start with BOM')
        return (object, consumed)

def getregentry():
    if False:
        i = 10
        return i + 15
    return codecs.CodecInfo(name='utf-16', encode=encode, decode=decode, incrementalencoder=IncrementalEncoder, incrementaldecoder=IncrementalDecoder, streamreader=StreamReader, streamwriter=StreamWriter)