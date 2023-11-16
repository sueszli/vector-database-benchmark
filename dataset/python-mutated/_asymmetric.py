from __future__ import annotations
import abc

class AsymmetricPadding(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        if False:
            return 10
        '\n        A string naming this padding (e.g. "PSS", "PKCS1").\n        '