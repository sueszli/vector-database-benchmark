from typing import Dict, Optional
import base58
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Base58_flickr(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            i = 10
            return i + 15
        '\n        Performs Base58 (Flickr) decoding\n        '
        FLICKR_ALPHABET = b'123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ'
        try:
            return base58.b58decode(ctext, alphabet=FLICKR_ALPHABET).decode('utf-8')
        except Exception:
            return None

    @staticmethod
    def priority() -> float:
        if False:
            for i in range(10):
                print('nop')
        return 0.05

    def __init__(self, config: Config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            i = 10
            return i + 15
        return None

    @staticmethod
    def getTarget() -> str:
        if False:
            return 10
        return 'base58_flickr'