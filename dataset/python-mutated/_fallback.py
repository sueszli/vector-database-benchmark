from pypdf._crypt_providers._base import CryptBase
from pypdf.errors import DependencyError
_DEPENDENCY_ERROR_STR = 'cryptography>=3.1 is required for AES algorithm'
crypt_provider = ('local_crypt_fallback', '0.0.0')

class CryptRC4(CryptBase):

    def __init__(self, key: bytes) -> None:
        if False:
            return 10
        self.s = bytearray(range(256))
        j = 0
        for i in range(256):
            j = (j + self.s[i] + key[i % len(key)]) % 256
            (self.s[i], self.s[j]) = (self.s[j], self.s[i])

    def encrypt(self, data: bytes) -> bytes:
        if False:
            while True:
                i = 10
        s = bytearray(self.s)
        out = [0 for _ in range(len(data))]
        (i, j) = (0, 0)
        for k in range(len(data)):
            i = (i + 1) % 256
            j = (j + s[i]) % 256
            (s[i], s[j]) = (s[j], s[i])
            x = s[(s[i] + s[j]) % 256]
            out[k] = data[k] ^ x
        return bytes(bytearray(out))

    def decrypt(self, data: bytes) -> bytes:
        if False:
            i = 10
            return i + 15
        return self.encrypt(data)

class CryptAES(CryptBase):

    def __init__(self, key: bytes) -> None:
        if False:
            while True:
                i = 10
        pass

    def encrypt(self, data: bytes) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        raise DependencyError(_DEPENDENCY_ERROR_STR)

    def decrypt(self, data: bytes) -> bytes:
        if False:
            print('Hello World!')
        raise DependencyError(_DEPENDENCY_ERROR_STR)

def rc4_encrypt(key: bytes, data: bytes) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    return CryptRC4(key).encrypt(data)

def rc4_decrypt(key: bytes, data: bytes) -> bytes:
    if False:
        while True:
            i = 10
    return CryptRC4(key).decrypt(data)

def aes_ecb_encrypt(key: bytes, data: bytes) -> bytes:
    if False:
        return 10
    raise DependencyError(_DEPENDENCY_ERROR_STR)

def aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    raise DependencyError(_DEPENDENCY_ERROR_STR)

def aes_cbc_encrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    if False:
        return 10
    raise DependencyError(_DEPENDENCY_ERROR_STR)

def aes_cbc_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    if False:
        while True:
            i = 10
    raise DependencyError(_DEPENDENCY_ERROR_STR)