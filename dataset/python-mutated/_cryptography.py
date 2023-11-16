import secrets
from cryptography import __version__
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers.algorithms import AES, ARC4
from cryptography.hazmat.primitives.ciphers.base import Cipher
from cryptography.hazmat.primitives.ciphers.modes import CBC, ECB
from pypdf._crypt_providers._base import CryptBase
crypt_provider = ('cryptography', __version__)

class CryptRC4(CryptBase):

    def __init__(self, key: bytes) -> None:
        if False:
            while True:
                i = 10
        self.cipher = Cipher(ARC4(key), mode=None)

    def encrypt(self, data: bytes) -> bytes:
        if False:
            i = 10
            return i + 15
        encryptor = self.cipher.encryptor()
        return encryptor.update(data) + encryptor.finalize()

    def decrypt(self, data: bytes) -> bytes:
        if False:
            return 10
        decryptor = self.cipher.decryptor()
        return decryptor.update(data) + decryptor.finalize()

class CryptAES(CryptBase):

    def __init__(self, key: bytes) -> None:
        if False:
            return 10
        self.alg = AES(key)

    def encrypt(self, data: bytes) -> bytes:
        if False:
            while True:
                i = 10
        iv = secrets.token_bytes(16)
        pad = padding.PKCS7(128).padder()
        data = pad.update(data) + pad.finalize()
        cipher = Cipher(self.alg, CBC(iv))
        encryptor = cipher.encryptor()
        return iv + encryptor.update(data) + encryptor.finalize()

    def decrypt(self, data: bytes) -> bytes:
        if False:
            i = 10
            return i + 15
        iv = data[:16]
        data = data[16:]
        if not data:
            return data
        if len(data) % 16 != 0:
            pad = padding.PKCS7(128).padder()
            data = pad.update(data) + pad.finalize()
        cipher = Cipher(self.alg, CBC(iv))
        decryptor = cipher.decryptor()
        d = decryptor.update(data) + decryptor.finalize()
        return d[:-d[-1]]

def rc4_encrypt(key: bytes, data: bytes) -> bytes:
    if False:
        i = 10
        return i + 15
    encryptor = Cipher(ARC4(key), mode=None).encryptor()
    return encryptor.update(data) + encryptor.finalize()

def rc4_decrypt(key: bytes, data: bytes) -> bytes:
    if False:
        print('Hello World!')
    decryptor = Cipher(ARC4(key), mode=None).decryptor()
    return decryptor.update(data) + decryptor.finalize()

def aes_ecb_encrypt(key: bytes, data: bytes) -> bytes:
    if False:
        while True:
            i = 10
    encryptor = Cipher(AES(key), mode=ECB()).encryptor()
    return encryptor.update(data) + encryptor.finalize()

def aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    if False:
        while True:
            i = 10
    decryptor = Cipher(AES(key), mode=ECB()).decryptor()
    return decryptor.update(data) + decryptor.finalize()

def aes_cbc_encrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    if False:
        i = 10
        return i + 15
    encryptor = Cipher(AES(key), mode=CBC(iv)).encryptor()
    return encryptor.update(data) + encryptor.finalize()

def aes_cbc_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    if False:
        i = 10
        return i + 15
    decryptor = Cipher(AES(key), mode=CBC(iv)).decryptor()
    return decryptor.update(data) + decryptor.finalize()