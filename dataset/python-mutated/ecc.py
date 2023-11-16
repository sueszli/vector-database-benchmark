from ctypes import *
ECCref_MAX_BITS = 512
ECCref_MAX_LEN = int((ECCref_MAX_BITS + 7) / 8)

class EncodeMixin:

    def encode(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

class ECCrefPublicKey(Structure, EncodeMixin):
    _fields_ = [('bits', c_uint), ('x', c_ubyte * ECCref_MAX_LEN), ('y', c_ubyte * ECCref_MAX_LEN)]

    def encode(self):
        if False:
            for i in range(10):
                print('nop')
        return bytes([4]) + bytes(self.x[32:]) + bytes(self.y[32:])

class ECCrefPrivateKey(Structure, EncodeMixin):
    _fields_ = [('bits', c_uint), ('K', c_ubyte * ECCref_MAX_LEN)]

    def encode(self):
        if False:
            while True:
                i = 10
        return bytes(self.K[32:])

class ECCCipherEncode(EncodeMixin):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.x = None
        self.y = None
        self.M = None
        self.C = None
        self.L = None

    def encode(self):
        if False:
            print('Hello World!')
        c1 = bytes(self.x[32:]) + bytes(self.y[32:])
        c2 = bytes(self.C[:self.L])
        c3 = bytes(self.M)
        return bytes([4]) + c1 + c2 + c3

def new_ecc_cipher_cla(length):
    if False:
        return 10
    _cache = {}
    cla_name = 'ECCCipher{}'.format(length)
    if _cache.__contains__(cla_name):
        return _cache[cla_name]
    else:
        cla = type(cla_name, (Structure, ECCCipherEncode), {'_fields_': [('x', c_ubyte * ECCref_MAX_LEN), ('y', c_ubyte * ECCref_MAX_LEN), ('M', c_ubyte * 32), ('L', c_uint), ('C', c_ubyte * length)]})
        _cache[cla_name] = cla
        return cla

class ECCKeyPair:

    def __init__(self, public_key, private_key):
        if False:
            print('Hello World!')
        self.public_key = public_key
        self.private_key = private_key