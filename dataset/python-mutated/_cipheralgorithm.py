from __future__ import annotations
import abc

class CipherAlgorithm(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        A string naming this mode (e.g. "AES", "Camellia").\n        '

    @property
    @abc.abstractmethod
    def key_sizes(self) -> frozenset[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Valid key sizes for this algorithm in bits\n        '

    @property
    @abc.abstractmethod
    def key_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        The size of the key being used as an integer in bits (e.g. 128, 256).\n        '

class BlockCipherAlgorithm(CipherAlgorithm):
    key: bytes

    @property
    @abc.abstractmethod
    def block_size(self) -> int:
        if False:
            while True:
                i = 10
        '\n        The size of a block as an integer in bits (e.g. 64, 128).\n        '