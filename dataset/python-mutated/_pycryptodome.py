import secrets
from Crypto import __version__
from Crypto.Cipher import AES, ARC4
from Crypto.Util.Padding import pad
from pypdf._crypt_providers._base import CryptBase
crypt_provider = ('pycryptodome', __version__)

class CryptRC4(CryptBase):

    def __init__(self, key: bytes) -> None:
        if False:
            i = 10
            return i + 15
        self.key = key

    def encrypt(self, data: bytes) -> bytes:
        if False:
            while True:
                i = 10
        return ARC4.ARC4Cipher(self.key).encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        if False:
            print('Hello World!')
        return ARC4.ARC4Cipher(self.key).decrypt(data)

class CryptAES(CryptBase):

    def __init__(self, key: bytes) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.key = key

    def encrypt(self, data: bytes) -> bytes:
        if False:
            while True:
                i = 10
        iv = secrets.token_bytes(16)
        data = pad(data, 16)
        aes = AES.new(self.key, AES.MODE_CBC, iv)
        return iv + aes.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        iv = data[:16]
        data = data[16:]
        if not data:
            return data
        if len(data) % 16 != 0:
            data = pad(data, 16)
        aes = AES.new(self.key, AES.MODE_CBC, iv)
        d = aes.decrypt(data)
        return d[:-d[-1]]

def rc4_encrypt(key: bytes, data: bytes) -> bytes:
    if False:
        i = 10
        return i + 15
    return ARC4.ARC4Cipher(key).encrypt(data)

def rc4_decrypt(key: bytes, data: bytes) -> bytes:
    if False:
        print('Hello World!')
    return ARC4.ARC4Cipher(key).decrypt(data)

def aes_ecb_encrypt(key: bytes, data: bytes) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    return AES.new(key, AES.MODE_ECB).encrypt(data)

def aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    return AES.new(key, AES.MODE_ECB).decrypt(data)

def aes_cbc_encrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    if False:
        while True:
            i = 10
    return AES.new(key, AES.MODE_CBC, iv).encrypt(data)

def aes_cbc_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    if False:
        print('Hello World!')
    return AES.new(key, AES.MODE_CBC, iv).decrypt(data)