import logging
from abc import ABC
from abc import abstractmethod
from jrnl.exception import JrnlException
from jrnl.messages import Message
from jrnl.messages import MsgStyle
from jrnl.messages import MsgText

class BaseEncryption(ABC):

    def __init__(self, journal_name: str, config: dict):
        if False:
            i = 10
            return i + 15
        logging.debug('start')
        self._encoding: str = 'utf-8'
        self._journal_name: str = journal_name
        self._config: dict = config

    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def encrypt(self, text: str) -> bytes:
        if False:
            print('Hello World!')
        logging.debug('encrypting')
        return self._encrypt(text)

    def decrypt(self, text: bytes) -> str:
        if False:
            print('Hello World!')
        logging.debug('decrypting')
        if (result := self._decrypt(text)) is None:
            raise JrnlException(Message(MsgText.DecryptionFailedGeneric, MsgStyle.ERROR))
        return result

    @abstractmethod
    def _encrypt(self, text: str) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        This is needed because self.decrypt might need\n        to perform actions (e.g. prompt for password)\n        before actually encrypting.\n        '
        pass

    @abstractmethod
    def _decrypt(self, text: bytes) -> str | None:
        if False:
            i = 10
            return i + 15
        '\n        This is needed because self.decrypt might need\n        to perform actions (e.g. prompt for password)\n        before actually decrypting.\n        '
        pass