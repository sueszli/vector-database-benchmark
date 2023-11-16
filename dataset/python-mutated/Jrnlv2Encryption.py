import base64
import logging
from cryptography.fernet import Fernet
from cryptography.fernet import InvalidToken
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from .BasePasswordEncryption import BasePasswordEncryption

class Jrnlv2Encryption(BasePasswordEncryption):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._salt: bytes = b'\xf2\xd5q\x0e\xc1\x8d.\xde\xdc\x8e6t\x89\x04\xce\xf8'
        self._key: bytes = b''
        super().__init__(*args, **kwargs)
        logging.debug('start')

    @property
    def password(self):
        if False:
            while True:
                i = 10
        return self._password

    @password.setter
    def password(self, value: str | None):
        if False:
            return 10
        self._password = value
        self._make_key()

    def _make_key(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._password is None:
            self._key = None
            return
        password = self.password.encode(self._encoding)
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=self._salt, iterations=100000, backend=default_backend())
        key = kdf.derive(password)
        self._key = base64.urlsafe_b64encode(key)

    def _encrypt(self, text: str) -> bytes:
        if False:
            while True:
                i = 10
        logging.debug('encrypting')
        return Fernet(self._key).encrypt(text.encode(self._encoding))

    def _decrypt(self, text: bytes) -> str | None:
        if False:
            while True:
                i = 10
        logging.debug('decrypting')
        try:
            return Fernet(self._key).decrypt(text).decode(self._encoding)
        except (InvalidToken, IndexError):
            return None