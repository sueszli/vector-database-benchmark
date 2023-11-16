import hashlib
import logging
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.ciphers import modes
from jrnl.encryption.BasePasswordEncryption import BasePasswordEncryption

class Jrnlv1Encryption(BasePasswordEncryption):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        logging.debug('start')

    def _encrypt(self, _: str) -> bytes:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def _decrypt(self, text: bytes) -> str | None:
        if False:
            return 10
        logging.debug('decrypting')
        (iv, cipher) = (text[:16], text[16:])
        password = self._password or ''
        decryption_key = hashlib.sha256(password.encode(self._encoding)).digest()
        decryptor = Cipher(algorithms.AES(decryption_key), modes.CBC(iv), default_backend()).decryptor()
        try:
            plain_padded = decryptor.update(cipher) + decryptor.finalize()
            if plain_padded[-1] in (' ', 32):
                return plain_padded.decode(self._encoding).rstrip(' ')
            else:
                unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
                plain = unpadder.update(plain_padded) + unpadder.finalize()
                return plain.decode(self._encoding)
        except ValueError:
            return None