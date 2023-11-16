import logging
from jrnl.encryption.BaseEncryption import BaseEncryption

class NoEncryption(BaseEncryption):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        logging.debug('start')

    def _encrypt(self, text: str) -> bytes:
        if False:
            return 10
        logging.debug('encrypting')
        return text.encode(self._encoding)

    def _decrypt(self, text: bytes) -> str:
        if False:
            while True:
                i = 10
        logging.debug('decrypting')
        return text.decode(self._encoding)